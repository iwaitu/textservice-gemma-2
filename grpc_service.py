import gc
import logging
import os
import time
from concurrent import futures

import grpc

from gemma_engine import TextEngine
import text_service_pb2
import text_service_pb2_grpc


LOGGER = logging.getLogger(__name__)


class TextGrpcServiceServicer(text_service_pb2_grpc.TextGrpcServiceServicer):
    def __init__(self):
        LOGGER.info("Loading ONNX models for gRPC service...")
        self.engine = TextEngine()
        self.embedding_model_name = self.engine.embedding_model_name
        self.reranker_model_name = self.engine.reranker_model_name

    def SimpleRerank(self, request, context):
        query = request.prompt
        documents = list(request.text_blocks)

        try:
            results = [
                text_service_pb2.RerankResult(
                    score=score,
                    text_block=doc,
                    index=index,
                )
                for index, doc, score in self.engine.rerank(query, documents)
            ]
            return text_service_pb2.SimpleRerankResponse(
                ranked_results=results,
                total_blocks=len(documents),
                query=query,
            )
        except Exception as exc:
            LOGGER.exception("SimpleRerank failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return text_service_pb2.SimpleRerankResponse()

    def EmbedText(self, request, context):
        texts = list(request.input)

        try:
            embeddings, text_lengths, total_tokens = self.engine.embed(texts)

            data = []
            for index, (embedding, text_length) in enumerate(zip(embeddings.tolist(), text_lengths)):
                data.append(
                    text_service_pb2.EmbeddingData(
                        object="embedding",
                        embedding=embedding,
                        index=index,
                        text_length=text_length,
                    )
                )

            usage = text_service_pb2.Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            )
            return text_service_pb2.EmbedTextResponse(
                data=data,
                model=self.embedding_model_name,
                object="list",
                usage=usage,
            )
        except Exception as exc:
            LOGGER.exception("EmbedText failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return text_service_pb2.EmbedTextResponse()

    def SplitTextIntoChunks(self, request, context):
        try:
            chunks = self.engine.split_text(
                request.input,
                request.chunk_size,
                request.overlap_size,
            )
            return text_service_pb2.SplitTextResponse(
                chunks=chunks,
                total_chunks=len(chunks),
                chunk_size=request.chunk_size,
                overlap_size=request.overlap_size,
                average_chunk_length=sum(len(chunk) for chunk in chunks) / max(len(chunks), 1),
            )
        except Exception as exc:
            LOGGER.exception("SplitTextIntoChunks failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return text_service_pb2.SplitTextResponse()

    def HealthCheck(self, request, context):
        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.1)
        return text_service_pb2.HealthCheckResponse(
            status="healthy",
            timestamp=time.time(),
            memory_usage_mb=str(memory_mb),
            cpu_percent=str(cpu_percent),
            pid=os.getpid(),
            models_loaded=text_service_pb2.ModelStatus(
                reranker=self.engine.reranker.session is not None,
                embedding=self.engine.embedding.session is not None,
            ),
        )

    def GetModelInfo(self, request, context):
        return text_service_pb2.ModelInfoResponse(
            reranker_model=text_service_pb2.ModelDetail(
                loaded=self.engine.reranker.session is not None,
                type="reranker",
                model_name=self.reranker_model_name,
            ),
            embedding_model=text_service_pb2.ModelDetail(
                loaded=self.engine.embedding.session is not None,
                type="embedding",
                model_name=self.embedding_model_name,
            ),
        )


def serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    port = os.environ.get("GRPC_PORT", "32688")
    servicer = TextGrpcServiceServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    text_service_pb2_grpc.add_TextGrpcServiceServicer_to_server(
        servicer,
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    LOGGER.info("gRPC server started on port %s", port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Received interrupt, stopping gRPC server...")
        server.stop(grace=5)
    finally:
        servicer.engine.close()
        gc.collect()
        LOGGER.info("gRPC server stopped, GPU resources released.")


if __name__ == "__main__":
    serve()
