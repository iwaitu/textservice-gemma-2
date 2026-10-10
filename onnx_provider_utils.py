import os
import platform

import onnxruntime as ort


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"

_PROVIDER_ALIASES = {
    "cpu": CPU_PROVIDER,
    "cpuexecutionprovider": CPU_PROVIDER,
    "cuda": CUDA_PROVIDER,
    "cudaexecutionprovider": CUDA_PROVIDER,
    "coreml": COREML_PROVIDER,
    "coremlexecutionprovider": COREML_PROVIDER,
    "auto": "auto",
}

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _normalize_provider_name(name):
    normalized = name.strip()
    alias = _PROVIDER_ALIASES.get(normalized.lower())
    return alias or normalized


def _parse_provider_chain(raw_value):
    providers = []
    for item in raw_value.split(","):
        normalized = _normalize_provider_name(item)
        if normalized and normalized != "auto":
            providers.append(normalized)
    return providers


def _env_flag(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in _TRUE_VALUES


def _build_cuda_provider_options():
    options = {
        # Avoid the default arena growth behavior from holding onto a much larger
        # GPU cache after occasional large requests.
        "arena_extend_strategy": os.environ.get(
            "ORT_CUDA_ARENA_EXTEND_STRATEGY",
            "kSameAsRequested",
        ).strip() or "kSameAsRequested",
    }

    device_id = os.environ.get("ORT_CUDA_DEVICE_ID", "").strip()
    if device_id:
        options["device_id"] = device_id

    gpu_mem_limit_mb = os.environ.get("ORT_CUDA_GPU_MEM_LIMIT_MB", "").strip()
    if gpu_mem_limit_mb:
        options["gpu_mem_limit"] = str(int(gpu_mem_limit_mb) * 1024 * 1024)

    if "ORT_CUDA_CUDNN_CONV_USE_MAX_WORKSPACE" in os.environ:
        options["cudnn_conv_use_max_workspace"] = (
            "1" if _env_flag("ORT_CUDA_CUDNN_CONV_USE_MAX_WORKSPACE") else "0"
        )

    if "ORT_CUDA_DO_COPY_IN_DEFAULT_STREAM" in os.environ:
        options["do_copy_in_default_stream"] = (
            "1" if _env_flag("ORT_CUDA_DO_COPY_IN_DEFAULT_STREAM", True) else "0"
        )

    return options


def choose_execution_providers():
    available = ort.get_available_providers()

    configured_chain = os.environ.get("ONNX_EXECUTION_PROVIDERS", "").strip()
    if configured_chain:
        requested = [provider for provider in _parse_provider_chain(configured_chain) if provider in available]
        if requested:
            if CPU_PROVIDER not in requested:
                requested.append(CPU_PROVIDER)
            return requested, f"configured via ONNX_EXECUTION_PROVIDERS={configured_chain}", available

    preferred_provider = os.environ.get("ONNX_PREFERRED_PROVIDER", os.environ.get("ONNX_PROVIDER", "auto")).strip()
    normalized_preferred = _normalize_provider_name(preferred_provider or "auto")
    if normalized_preferred != "auto":
        if normalized_preferred in available:
            if normalized_preferred == CPU_PROVIDER:
                return [CPU_PROVIDER], f"preferred provider {normalized_preferred} from environment", available
            return [normalized_preferred, CPU_PROVIDER], f"preferred provider {normalized_preferred} from environment", available
        return [CPU_PROVIDER], f"preferred provider {normalized_preferred} unavailable, falling back to CPU", available

    if platform.system() == "Darwin" and COREML_PROVIDER in available:
        return [COREML_PROVIDER, CPU_PROVIDER], "auto-detected macOS CoreML support", available

    if CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER], "auto-detected CUDA support", available

    return [CPU_PROVIDER], "defaulted to CPUExecutionProvider", available


def build_provider_chain(providers):
    configured = []
    for provider in providers:
        if provider == CUDA_PROVIDER:
            configured.append((CUDA_PROVIDER, _build_cuda_provider_options()))
        else:
            configured.append(provider)
    return configured


def provider_chain_to_string(providers):
    parts = []
    for provider in providers:
        if isinstance(provider, tuple):
            name, options = provider
            if options:
                parts.append(f"{name}{options}")
            else:
                parts.append(name)
        else:
            parts.append(provider)
    return " -> ".join(parts)
