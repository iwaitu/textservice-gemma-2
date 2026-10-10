"""从 Hugging Face 下载 EmbeddingGemma 2 的 ONNX 文本模型与视觉编码器（不下载音频编码器）。

默认同时下载视觉编码器（图片向量需要）；只做文本时可加 --no-vision。

用法:
    python tools/download_model.py                      # fp32（默认，推荐 GPU / 精度优先）
    python tools/download_model.py --variant quantized  # int8，CPU 部署体积更小
    python tools/download_model.py --no-vision          # 只下载文本模型
"""

import argparse
import os

from huggingface_hub import snapshot_download


REPO_ID = "onnx-community/embeddinggemma-2-ONNX"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Models", "embeddinggemma-2-onnx")

# variant -> onnx 文件名前缀
VARIANTS = {
    "fp32": "model",
    "fp16": "model_fp16",
    "quantized": "model_quantized",
    "q4": "model_q4",
    "q4f16": "model_q4f16",
}

METADATA_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="fp32")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--no-vision", action="store_true", help="不下载视觉编码器（关闭图片向量）")
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = VARIANTS[args.variant]
    patterns = METADATA_FILES + [f"onnx/{prefix}.onnx", f"onnx/{prefix}.onnx_data*"]
    vision_prefix = prefix.replace("model", "vision_encoder", 1)
    if not args.no_vision:
        patterns += [f"onnx/{vision_prefix}.onnx", f"onnx/{vision_prefix}.onnx_data*"]
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        allow_patterns=patterns,
        local_dir=args.output_dir,
    )
    model_path = os.path.join(path, "onnx", f"{prefix}.onnx")
    print(f"Downloaded {REPO_ID} ({args.variant}) to {path}")
    if prefix != "model":
        print(f"Set EMBEDDING_MODEL_PATH={model_path} to use this variant.")
        if not args.no_vision:
            print(f"Set VISION_MODEL_PATH={os.path.join(path, 'onnx', vision_prefix + '.onnx')} as well.")
    if args.variant in {"fp16", "q4f16"}:
        print("Warning: EmbeddingGemma 2 activations can overflow float16; validate quality before production use.")


if __name__ == "__main__":
    main()
