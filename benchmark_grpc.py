import argparse
import statistics
import sys
import time

import grpc

import text_service_pb2
import text_service_pb2_grpc


RERANK_ACCURACY_CASES = [
    {
        "name": "capital_of_china",
        "query": "What is the capital of China?",
        "documents": [
            "Gravity is a force that attracts two bodies towards each other.",
            "Beijing is the capital of China.",
            "Paris is the capital of France.",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        ],
        "relevant_indexes": [1],
    },
    {
        "name": "python_creator",
        "query": "Who created the Python programming language?",
        "documents": [
            "Java was originally developed by James Gosling at Sun Microsystems.",
            "Python is a popular programming language created by Guido van Rossum.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "C++ was developed by Bjarne Stroustrup.",
        ],
        "relevant_indexes": [1],
    },
    {
        "name": "largest_planet",
        "query": "Which planet is the largest in the solar system?",
        "documents": [
            "Mars is often called the Red Planet because of its reddish appearance.",
            "Jupiter is the largest planet in the solar system.",
            "Saturn is known for its ring system.",
            "Mercury is the closest planet to the Sun.",
        ],
        "relevant_indexes": [1],
    },
    {
        "name": "http_status_404",
        "query": "What does HTTP status code 404 mean?",
        "documents": [
            "HTTP 404 means the requested resource could not be found on the server.",
            "HTTP 500 indicates an internal server error.",
            "HTTP 301 is used for permanent redirects.",
            "A database index can improve query performance.",
        ],
        "relevant_indexes": [0],
    },
    {
        "name": "photosynthesis",
        "query": "What is photosynthesis?",
        "documents": [
            "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "Cellular respiration releases stored energy from glucose.",
            "Mitosis is a process of cell division.",
            "The Amazon rainforest contains high biodiversity.",
        ],
        "relevant_indexes": [0],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark EmbeddingGemma2 Text Service gRPC endpoints")
    parser.add_argument("--target", default="127.0.0.1:32688", help="gRPC service target")
    parser.add_argument("--host", default=None, help="Deprecated: gRPC service host")
    parser.add_argument("--port", type=int, default=None, help="Deprecated: gRPC service port")
    parser.add_argument("--runs", type=int, default=6, help="Number of benchmark runs")
    parser.add_argument("--embed-count", type=int, default=32, help="Number of texts for embedding benchmark")
    parser.add_argument("--rerank-count", type=int, default=50, help="Number of documents for rerank benchmark")
    parser.add_argument("--timeout", type=int, default=180, help="RPC timeout in seconds")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip curated rerank accuracy cases")
    return parser.parse_args()


def percentile(values, ratio):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * ratio
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def benchmark_rpc(call_fn, payload_size, runs):
    latencies = []
    sample = None

    for index in range(runs):
        started = time.perf_counter()
        response = call_fn()
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(round(elapsed_ms, 2))
        if index == 0:
            sample = response

    steady = latencies[1:] if len(latencies) > 1 else latencies
    steady_avg = statistics.mean(steady)
    return {
        "payload_size": payload_size,
        "runs_ms": latencies,
        "warmup_ms": latencies[0],
        "steady_min_ms": round(min(steady), 2),
        "steady_max_ms": round(max(steady), 2),
        "steady_avg_ms": round(steady_avg, 2),
        "steady_median_ms": round(statistics.median(steady), 2),
        "steady_p95_ms": round(percentile(steady, 0.95), 2),
        "throughput_per_sec": round(payload_size / (steady_avg / 1000), 2),
        "sample": sample,
    }


def build_target(args):
    if args.host is not None or args.port is not None:
        host = args.host or "127.0.0.1"
        port = args.port or 32688
        return f"{host}:{port}"
    return args.target


def evaluate_rerank_accuracy(stub, timeout):
    case_results = []

    for case in RERANK_ACCURACY_CASES:
        response = stub.SimpleRerank(
            text_service_pb2.SimpleRerankRequest(
                prompt=case["query"],
                text_blocks=case["documents"],
            ),
            timeout=timeout,
        )
        ranked_indexes = [item.index for item in response.ranked_results]

        first_relevant_rank = None
        for rank, index in enumerate(ranked_indexes, start=1):
            if index in case["relevant_indexes"]:
                first_relevant_rank = rank
                break

        top1_hit = bool(ranked_indexes and ranked_indexes[0] in case["relevant_indexes"])
        top3_hit = any(index in case["relevant_indexes"] for index in ranked_indexes[:3])
        reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        top_result = response.ranked_results[0]

        case_results.append(
            {
                "name": case["name"],
                "expected_indexes": case["relevant_indexes"],
                "ranked_indexes": ranked_indexes[:5],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "first_relevant_rank": first_relevant_rank,
                "mrr": round(reciprocal_rank, 4),
                "top_result": {
                    "index": top_result.index,
                    "score": top_result.score,
                    "text_block": top_result.text_block,
                },
            }
        )

    top1_accuracy = sum(1 for item in case_results if item["top1_hit"]) / len(case_results)
    top3_accuracy = sum(1 for item in case_results if item["top3_hit"]) / len(case_results)
    mean_reciprocal_rank = sum(item["mrr"] for item in case_results) / len(case_results)

    return {
        "case_count": len(case_results),
        "top1_accuracy": round(top1_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "mrr": round(mean_reciprocal_rank, 4),
        "cases": case_results,
    }


def main():
    args = parse_args()
    target = build_target(args)

    with grpc.insecure_channel(target) as channel:
        stub = text_service_pb2_grpc.TextGrpcServiceStub(channel)

        health = stub.HealthCheck(text_service_pb2.HealthCheckRequest(), timeout=args.timeout)
        health_body = {
            "status": health.status,
            "pid": health.pid,
            "memory_usage_mb": health.memory_usage_mb,
            "cpu_percent": health.cpu_percent,
            "models_loaded": {
                "embedding": health.models_loaded.embedding,
                "reranker": health.models_loaded.reranker,
            },
        }
        print("health", 200, health_body)

        model_info = stub.GetModelInfo(text_service_pb2.ModelInfoRequest(), timeout=args.timeout)
        print(
            "model_info",
            {
                "embedding_model": model_info.embedding_model.model_name,
                "reranker_model": model_info.reranker_model.model_name,
            },
        )

        embed_request = text_service_pb2.EmbedTextRequest(
            input=[
                f"Document {index}: Beijing is the capital of China. This sample text is used for embedding benchmark run {index}."
                for index in range(args.embed_count)
            ]
        )
        rerank_request = text_service_pb2.SimpleRerankRequest(
            prompt="What is the capital of China?",
            text_blocks=[
                (
                    f"Candidate {index}: Beijing is the capital of China and a major city with historical sites."
                    if index % 10 == 0
                    else f"Candidate {index}: This is a generic unrelated sentence used to simulate rerank candidates."
                )
                for index in range(args.rerank_count)
            ],
        )

        embed_result = benchmark_rpc(
            lambda: stub.EmbedText(embed_request, timeout=args.timeout),
            args.embed_count,
            args.runs,
        )
        rerank_result = benchmark_rpc(
            lambda: stub.SimpleRerank(rerank_request, timeout=args.timeout),
            args.rerank_count,
            args.runs,
        )

        print("embed_result", {key: value for key, value in embed_result.items() if key != "sample"})
        print("embed_dimension", len(embed_result["sample"].data[0].embedding))
        print("embed_usage_total_tokens", embed_result["sample"].usage.total_tokens)

        print("rerank_result", {key: value for key, value in rerank_result.items() if key != "sample"})
        print(
            "rerank_top3",
            [
                {
                    "index": item.index,
                    "score": item.score,
                    "text_block": item.text_block,
                }
                for item in rerank_result["sample"].ranked_results[:3]
            ],
        )

        if not args.skip_accuracy:
            accuracy_result = evaluate_rerank_accuracy(stub, args.timeout)
            print(
                "rerank_accuracy_summary",
                {key: value for key, value in accuracy_result.items() if key != "cases"},
            )
            for case in accuracy_result["cases"]:
                print("rerank_accuracy_case", case)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"benchmark_failed error={exc}", file=sys.stderr)
        sys.exit(1)