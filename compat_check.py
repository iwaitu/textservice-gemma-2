"""接口兼容性检查：验证 REST / gRPC 的请求与响应结构与 TextServiceQwen3 一致。

用法:
    python compat_check.py --grpc 127.0.0.1:32688
    python compat_check.py --rest http://127.0.0.1:8000
    # 同时与线上 TextServiceQwen3 对比响应字段结构（向量维度与分数除外）:
    python compat_check.py --grpc 127.0.0.1:32688 --reference-grpc qwen3-host:32688
"""

import argparse
import sys

import grpc
from google.protobuf.json_format import MessageToDict
import requests

import text_service_pb2
import text_service_pb2_grpc


QUERY = "What is the capital of China?"
DOCUMENTS = [
    "Gravity is a force that attracts two bodies towards each other.",
    "Beijing is the capital of China.",
    "Paris is the capital of France.",
    "北京是中华人民共和国的首都。",
]
TEXTS = ["Hello world", "你好世界", "Beijing is the capital of China."]
LONG_TEXT = "EmbeddingGemma 2 是一个多模态嵌入模型。" * 40

FAILURES = []


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def shape_of(value):
    """把 JSON 值抽象成结构（只保留键名与类型），用于跨服务对比。"""
    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [shape_of(value[0])] if value else []
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def grpc_calls(target):
    channel = grpc.insecure_channel(target)
    stub = text_service_pb2_grpc.TextGrpcServiceStub(channel)
    opts = {"timeout": 120}
    results = {
        "HealthCheck": stub.HealthCheck(text_service_pb2.HealthCheckRequest(), **opts),
        "GetModelInfo": stub.GetModelInfo(text_service_pb2.ModelInfoRequest(), **opts),
        "EmbedText": stub.EmbedText(text_service_pb2.EmbedTextRequest(input=TEXTS), **opts),
        "EmbedTextEmpty": stub.EmbedText(text_service_pb2.EmbedTextRequest(input=[]), **opts),
        "SimpleRerank": stub.SimpleRerank(
            text_service_pb2.SimpleRerankRequest(prompt=QUERY, text_blocks=DOCUMENTS), **opts
        ),
        "SimpleRerankEmpty": stub.SimpleRerank(
            text_service_pb2.SimpleRerankRequest(prompt=QUERY, text_blocks=[]), **opts
        ),
        "SplitTextIntoChunks": stub.SplitTextIntoChunks(
            text_service_pb2.SplitTextRequest(input=LONG_TEXT, chunk_size=64, overlap_size=8), **opts
        ),
    }
    try:
        stub.SplitTextIntoChunks(
            text_service_pb2.SplitTextRequest(input="abc", chunk_size=10, overlap_size=10), **opts
        )
        results["SplitTextInvalid"] = None
    except grpc.RpcError as exc:
        results["SplitTextInvalid"] = exc.code()
    channel.close()
    return results


def verify_grpc(target):
    print(f"gRPC {target}")
    r = grpc_calls(target)

    check(r["HealthCheck"].status == "healthy", "HealthCheck.status == healthy")
    check(r["HealthCheck"].models_loaded.embedding and r["HealthCheck"].models_loaded.reranker,
          "HealthCheck.models_loaded embedding & reranker")
    info = r["GetModelInfo"]
    check(info.embedding_model.type == "embedding" and info.embedding_model.loaded, "GetModelInfo.embedding_model")
    check(info.reranker_model.type == "reranker" and info.reranker_model.loaded, "GetModelInfo.reranker_model")

    embed = r["EmbedText"]
    check(embed.object == "list" and embed.model != "", "EmbedText.object/model")
    check(len(embed.data) == len(TEXTS), "EmbedText.data length == inputs")
    check([item.index for item in embed.data] == list(range(len(TEXTS))), "EmbedText.data index order")
    check(all(item.object == "embedding" for item in embed.data), "EmbedText.data.object == embedding")
    dims = {len(item.embedding) for item in embed.data}
    check(len(dims) == 1, f"EmbedText consistent dimension {dims}")
    norms = [sum(v * v for v in item.embedding) ** 0.5 for item in embed.data]
    check(all(abs(n - 1.0) < 1e-3 for n in norms), "EmbedText vectors are L2 normalized")
    check(all(item.text_length > 0 for item in embed.data), "EmbedText.text_length > 0")
    check(embed.usage.total_tokens == embed.usage.prompt_tokens > 0, "EmbedText.usage")
    check(len(r["EmbedTextEmpty"].data) == 0, "EmbedText empty input -> empty data")

    rerank = r["SimpleRerank"]
    check(rerank.total_blocks == len(DOCUMENTS) and rerank.query == QUERY, "SimpleRerank.total_blocks/query")
    scores = [item.score for item in rerank.ranked_results]
    check(scores == sorted(scores, reverse=True), "SimpleRerank sorted by score desc")
    check(all(0.0 <= s <= 1.0 for s in scores), "SimpleRerank scores in [0, 1]")
    check(sorted(item.index for item in rerank.ranked_results) == list(range(len(DOCUMENTS))),
          "SimpleRerank indexes cover all inputs")
    check(all(DOCUMENTS[item.index] == item.text_block for item in rerank.ranked_results),
          "SimpleRerank text_block matches index")
    check(rerank.ranked_results[0].index in (1, 3), "SimpleRerank top1 is a relevant document")
    check(len(r["SimpleRerankEmpty"].ranked_results) == 0, "SimpleRerank empty input")

    split = r["SplitTextIntoChunks"]
    check(split.total_chunks == len(split.chunks) > 1, "SplitTextIntoChunks chunk count")
    check(split.chunk_size == 64 and split.overlap_size == 8, "SplitTextIntoChunks echoes sizes")
    check(split.average_chunk_length > 0, "SplitTextIntoChunks.average_chunk_length")
    check(r["SplitTextInvalid"] == grpc.StatusCode.INTERNAL, "SplitTextIntoChunks invalid overlap -> INTERNAL")
    print(f"  info: embedding model={embed.model} dim={next(iter(dims))} "
          f"reranker model={info.reranker_model.model_name}")
    print(f"  info: rerank={[(item.index, round(item.score, 4)) for item in rerank.ranked_results]}")
    return r


def rest_calls(base_url):
    s = requests.Session()
    out = {
        "health": s.get(f"{base_url}/health", timeout=30),
        "embed": s.post(f"{base_url}/embed_text", json={"input": TEXTS}, timeout=120),
        "embed_empty": s.post(f"{base_url}/embed_text", json={"input": []}, timeout=120),
        "rerank": s.post(f"{base_url}/rerank", json={"query": QUERY, "documents": DOCUMENTS}, timeout=120),
        "rerank_instruction": s.post(
            f"{base_url}/rerank",
            json={
                "query": QUERY,
                "documents": DOCUMENTS,
                "instruction": "Given a web search query, retrieve relevant passages that answer the query",
            },
            timeout=120,
        ),
        "split": s.post(
            f"{base_url}/split_text",
            json={"text": LONG_TEXT, "chunksize": 64, "overlap_size": 8},
            timeout=120,
        ),
        "split_invalid": s.post(
            f"{base_url}/split_text",
            json={"text": "abc", "chunksize": 10, "overlap_size": 10},
            timeout=120,
        ),
        "bad_request": s.post(f"{base_url}/embed_text", json={"inputs": TEXTS}, timeout=30),
    }
    return out


def verify_rest(base_url):
    print(f"REST {base_url}")
    r = rest_calls(base_url)

    health = r["health"].json()
    check(set(health) == {"status", "embedding_model", "reranker_model", "device", "service"}, "GET /health keys")
    check(health["status"] == "healthy", "GET /health status")

    embed = r["embed"].json()
    check(set(embed) == {"data", "model", "object", "usage"}, "POST /embed_text keys")
    check(set(embed["data"][0]) == {"object", "embedding", "index", "tokens_length"}, "embed data item keys")
    check(set(embed["usage"]) == {"prompt_tokens", "total_tokens"}, "embed usage keys")
    check([d["index"] for d in embed["data"]] == list(range(len(TEXTS))), "embed index order")
    check(r["embed_empty"].status_code == 200 and r["embed_empty"].json()["data"] == [], "embed empty input")

    for key in ("rerank", "rerank_instruction"):
        body = r[key].json()
        check(set(body) == {"results", "model"}, f"POST /rerank keys ({key})")
        check(set(body["results"][0]) == {"document", "score", "index"}, f"rerank result keys ({key})")
        scores = [item["score"] for item in body["results"]]
        check(scores == sorted(scores, reverse=True), f"rerank sorted desc ({key})")
        check(body["results"][0]["index"] in (1, 3), f"rerank top1 relevant ({key})")

    split = r["split"].json()
    check(set(split) == {"chunks", "total_chunks", "model"}, "POST /split_text keys")
    check(split["total_chunks"] == len(split["chunks"]) > 1, "split chunk count")
    check(r["split_invalid"].status_code == 500, "split invalid overlap -> 500")
    check(r["bad_request"].status_code == 422, "invalid body -> 422")
    print(f"  info: embedding model={embed['model']} dim={len(embed['data'][0]['embedding'])}")
    print(f"  info: rerank={[(i['index'], round(i['score'], 4)) for i in r['rerank'].json()['results']]}")
    return r


def compare_grpc_shapes(candidate, reference):
    print("gRPC structure diff vs reference")
    for name, message in candidate.items():
        if message is None or isinstance(message, grpc.StatusCode):
            check(message == reference.get(name), f"{name} status matches reference")
            continue
        left = shape_of(MessageToDict(message, always_print_fields_with_no_presence=True))
        right = shape_of(MessageToDict(reference[name], always_print_fields_with_no_presence=True))
        check(left == right, f"{name} response structure matches reference")


def compare_rest_shapes(candidate, reference):
    print("REST structure diff vs reference")
    for name, response in candidate.items():
        ref = reference[name]
        check(response.status_code == ref.status_code, f"{name} status {response.status_code} == {ref.status_code}")
        if response.status_code == 200:
            left = shape_of(response.json())
            right = shape_of(ref.json())
            if name == "health":
                left.pop("device", None), right.pop("device", None)
            check(left == right, f"{name} response structure matches reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grpc", help="gRPC target of textservice-gemma-2, e.g. 127.0.0.1:32688")
    parser.add_argument("--rest", help="REST base url of textservice-gemma-2, e.g. http://127.0.0.1:8000")
    parser.add_argument("--reference-grpc", help="gRPC target of TextServiceQwen3 for structure comparison")
    parser.add_argument("--reference-rest", help="REST base url of TextServiceQwen3 for structure comparison")
    args = parser.parse_args()
    if not args.grpc and not args.rest:
        parser.error("specify --grpc and/or --rest")

    if args.grpc:
        candidate = verify_grpc(args.grpc)
        if args.reference_grpc:
            compare_grpc_shapes(candidate, grpc_calls(args.reference_grpc))
    if args.rest:
        candidate = verify_rest(args.rest.rstrip("/"))
        if args.reference_rest:
            compare_rest_shapes(candidate, rest_calls(args.reference_rest.rstrip("/")))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        sys.exit(1)
    print("All compatibility checks passed")


if __name__ == "__main__":
    main()
