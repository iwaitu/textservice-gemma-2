import argparse
import statistics
import subprocess
import sys
import time

import grpc

import text_service_pb2
import text_service_pb2_grpc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe gRPC service GPU memory behavior with repeated requests.",
    )
    parser.add_argument("--target", default="127.0.0.1:32688", help="gRPC service target")
    parser.add_argument("--timeout", type=int, default=300, help="RPC timeout in seconds")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for nvidia-smi sampling")
    parser.add_argument("--rounds", type=int, default=8, help="Number of stress rounds")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Seconds to wait after each round")
    parser.add_argument("--embed-count", type=int, default=24, help="Texts per embedding request")
    parser.add_argument("--embed-repeat", type=int, default=320, help="Text length multiplier for embedding")
    parser.add_argument("--rerank-count", type=int, default=40, help="Documents per rerank request")
    parser.add_argument("--rerank-repeat", type=int, default=220, help="Text length multiplier for rerank")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding requests")
    parser.add_argument("--skip-rerank", action="store_true", help="Skip rerank requests")
    return parser.parse_args()


def query_gpu_memory(gpu_index):
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    rows = []
    for line in output.strip().splitlines():
        index, name, used, total = [item.strip() for item in line.split(",", 3)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": int(used),
                "memory_total_mb": int(total),
            }
        )
    for row in rows:
        if row["index"] == gpu_index:
            return row
    raise RuntimeError(f"GPU index {gpu_index} not found in nvidia-smi output")


def build_embed_request(embed_count, repeat):
    texts = []
    repeated_segment = (
        "Beijing is the capital of China. "
        "This sentence is repeated to expand the sequence length for embedding stress. "
    ) * repeat
    for index in range(embed_count):
        texts.append(f"Embedding sample {index}. {repeated_segment}")
    return text_service_pb2.EmbedTextRequest(input=texts)


def build_rerank_request(rerank_count, repeat):
    query = (
        "Which city is the capital of China and what supporting evidence is present in the passage?"
    )
    positive = (
        "Beijing is the capital of China. "
        "It is the political center of the country. "
    ) * repeat
    negative = (
        "This paragraph talks about rivers, mountains, and unrelated geography. "
        "It does not answer the capital city question. "
    ) * repeat
    documents = []
    for index in range(rerank_count):
        if index % 8 == 0:
            documents.append(f"Relevant candidate {index}. {positive}")
        else:
            documents.append(f"Irrelevant candidate {index}. {negative}")
    return text_service_pb2.SimpleRerankRequest(prompt=query, text_blocks=documents)


def call_health(stub, timeout):
    health = stub.HealthCheck(text_service_pb2.HealthCheckRequest(), timeout=timeout)
    return {
        "status": health.status,
        "pid": health.pid,
        "rss_mb": float(health.memory_usage_mb),
        "cpu_percent": float(health.cpu_percent),
        "models_loaded": {
            "embedding": health.models_loaded.embedding,
            "reranker": health.models_loaded.reranker,
        },
    }


def summarize_series(name, values):
    if not values:
        return None
    return {
        "name": name,
        "min": min(values),
        "max": max(values),
        "avg": round(statistics.mean(values), 2),
        "final": values[-1],
    }


def main():
    args = parse_args()
    embed_request = None if args.skip_embed else build_embed_request(args.embed_count, args.embed_repeat)
    rerank_request = None if args.skip_rerank else build_rerank_request(args.rerank_count, args.rerank_repeat)

    with grpc.insecure_channel(args.target) as channel:
        stub = text_service_pb2_grpc.TextGrpcServiceStub(channel)
        initial_gpu = query_gpu_memory(args.gpu_index)
        initial_health = call_health(stub, args.timeout)

        print("probe.target", args.target)
        print("probe.gpu", initial_gpu)
        print("probe.health.initial", initial_health)

        gpu_after_round = []
        gpu_after_cooldown = []
        rss_after_round = []
        embed_latencies_ms = []
        rerank_latencies_ms = []
        failure = None

        for round_index in range(1, args.rounds + 1):
            before_gpu = query_gpu_memory(args.gpu_index)
            try:
                started = time.perf_counter()
                if embed_request is not None:
                    stub.EmbedText(embed_request, timeout=args.timeout)
                    embed_latencies_ms.append(round((time.perf_counter() - started) * 1000, 2))
                    started = time.perf_counter()
                if rerank_request is not None:
                    stub.SimpleRerank(rerank_request, timeout=args.timeout)
                    rerank_latencies_ms.append(round((time.perf_counter() - started) * 1000, 2))
            except grpc.RpcError as exc:
                after_gpu = query_gpu_memory(args.gpu_index)
                try:
                    after_health = call_health(stub, args.timeout)
                except Exception:
                    after_health = None
                failure = {
                    "round": round_index,
                    "code": exc.code().name if exc.code() is not None else None,
                    "details": exc.details(),
                    "gpu_before_mb": before_gpu["memory_used_mb"],
                    "gpu_after_mb": after_gpu["memory_used_mb"],
                    "health_after_failure": after_health,
                }
                print("round_failure", failure)
                break

            after_gpu = query_gpu_memory(args.gpu_index)
            after_health = call_health(stub, args.timeout)
            gpu_after_round.append(after_gpu["memory_used_mb"])
            rss_after_round.append(after_health["rss_mb"])

            time.sleep(args.cooldown)
            cooldown_gpu = query_gpu_memory(args.gpu_index)
            gpu_after_cooldown.append(cooldown_gpu["memory_used_mb"])

            print(
                "round",
                {
                    "index": round_index,
                    "gpu_before_mb": before_gpu["memory_used_mb"],
                    "gpu_after_mb": after_gpu["memory_used_mb"],
                    "gpu_after_cooldown_mb": cooldown_gpu["memory_used_mb"],
                    "gpu_delta_from_initial_mb": cooldown_gpu["memory_used_mb"] - initial_gpu["memory_used_mb"],
                    "rss_mb": round(after_health["rss_mb"], 2),
                    "embed_latency_ms": embed_latencies_ms[-1] if embed_latencies_ms else None,
                    "rerank_latency_ms": rerank_latencies_ms[-1] if rerank_latencies_ms else None,
                },
            )

        final_gpu = query_gpu_memory(args.gpu_index)
        try:
            final_health = call_health(stub, args.timeout)
        except Exception:
            final_health = initial_health
        print(
            "summary",
            {
                "initial_gpu_mb": initial_gpu["memory_used_mb"],
                "final_gpu_mb": final_gpu["memory_used_mb"],
                "peak_gpu_after_round_mb": max(gpu_after_round) if gpu_after_round else initial_gpu["memory_used_mb"],
                "peak_gpu_after_cooldown_mb": max(gpu_after_cooldown) if gpu_after_cooldown else initial_gpu["memory_used_mb"],
                "final_gpu_delta_mb": final_gpu["memory_used_mb"] - initial_gpu["memory_used_mb"],
                "initial_rss_mb": round(initial_health["rss_mb"], 2),
                "final_rss_mb": round(final_health["rss_mb"], 2),
                "final_rss_delta_mb": round(final_health["rss_mb"] - initial_health["rss_mb"], 2),
                "embed_latency_ms": summarize_series("embed_latency_ms", embed_latencies_ms),
                "rerank_latency_ms": summarize_series("rerank_latency_ms", rerank_latencies_ms),
                "gpu_after_round_mb": summarize_series("gpu_after_round_mb", gpu_after_round),
                "gpu_after_cooldown_mb": summarize_series("gpu_after_cooldown_mb", gpu_after_cooldown),
                "rss_after_round_mb": summarize_series("rss_after_round_mb", rss_after_round),
                "failure": failure,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"probe.failed error={exc}", file=sys.stderr)
        sys.exit(1)
