"""图片解码与 EmbeddingGemma 2（Gemma4 vision）预处理。

预处理与 transformers 的 Gemma4ImageProcessorPil 一致：
保持宽高比缩放（BICUBIC）-> 像素 /255 -> 16x16 patch 切分 -> 补齐到 max_soft_tokens * 3^2 个 patch。
"""

import base64
import binascii
import io
import ipaddress
import math
import os
import socket
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3

IMAGE_MAX_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", str(20 * 1024 * 1024)))
IMAGE_MAX_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", str(64 * 1024 * 1024)))
# 允许 REST /embed_image 通过 http(s) URL 拉取图片。默认关闭，避免服务被用来访问内网地址（SSRF）。
ALLOW_IMAGE_URLS = os.environ.get("ALLOW_IMAGE_URLS", "").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_PRIVATE_IMAGE_URLS = os.environ.get("ALLOW_PRIVATE_IMAGE_URLS", "").strip().lower() in {
    "1", "true", "yes", "on",
}
IMAGE_URL_TIMEOUT = float(os.environ.get("IMAGE_URL_TIMEOUT", "10"))

Image.MAX_IMAGE_PIXELS = IMAGE_MAX_PIXELS


def get_aspect_ratio_preserving_size(height, width, max_patches):
    """返回不超过 max_patches 个 patch、且边长是 48 的倍数的最大目标尺寸。"""
    target_px = max_patches * (PATCH_SIZE ** 2)
    factor = math.sqrt(target_px / (height * width))
    side_mult = POOLING_KERNEL_SIZE * PATCH_SIZE

    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError("Image is too small to be resized to a valid patch grid")

    max_side_length = (max_patches // POOLING_KERNEL_SIZE ** 2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    return target_height, target_width


def preprocess_image(image, max_soft_tokens):
    """PIL 图片 -> (patches[max_patches, 768], positions[max_patches, 2], 实际 soft token 数)。"""
    if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
        raise ValueError(f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}")
    max_patches = max_soft_tokens * POOLING_KERNEL_SIZE ** 2

    image = image.convert("RGB")
    target_height, target_width = get_aspect_ratio_preserving_size(image.height, image.width, max_patches)
    if (target_height, target_width) != (image.height, image.width):
        image = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)

    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (C, H, W)
    channels, height, width = pixels.shape
    patch_rows, patch_cols = height // PATCH_SIZE, width // PATCH_SIZE
    patches = (
        pixels.reshape(channels, patch_rows, PATCH_SIZE, patch_cols, PATCH_SIZE)
        .transpose(1, 3, 2, 4, 0)
        .reshape(patch_rows * patch_cols, -1)
    )
    grid_x, grid_y = np.meshgrid(np.arange(patch_cols), np.arange(patch_rows), indexing="xy")
    positions = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.int64)

    num_soft_tokens = patches.shape[0] // POOLING_KERNEL_SIZE ** 2
    padding = max_patches - patches.shape[0]
    if padding > 0:
        patches = np.pad(patches, ((0, padding), (0, 0)), constant_values=0)
        positions = np.pad(positions, ((0, padding), (0, 0)), constant_values=-1)
    return patches.astype(np.float32), positions, num_soft_tokens


def decode_image_bytes(data):
    if not data:
        raise ValueError("Empty image data")
    if len(data) > IMAGE_MAX_BYTES:
        raise ValueError(f"Image exceeds IMAGE_MAX_BYTES={IMAGE_MAX_BYTES}")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image exceeds IMAGE_MAX_PIXELS={IMAGE_MAX_PIXELS}") from exc
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported or corrupt image: cannot identify image format") from exc
    except Exception as exc:
        raise ValueError(f"Unsupported or corrupt image: {exc}") from exc
    # 按 EXIF 方向摆正（手机照片常见）
    return ImageOps.exif_transpose(image)


def _check_public_host(hostname):
    for info in socket.getaddrinfo(hostname, None):
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ValueError(f"Image URL host {hostname} resolves to a non-public address")


def _fetch_url(url):
    import requests

    parsed = urlparse(url)
    if not ALLOW_IMAGE_URLS:
        raise ValueError("Image URLs are disabled; set ALLOW_IMAGE_URLS=1 or send base64 data")
    if not ALLOW_PRIVATE_IMAGE_URLS:
        _check_public_host(parsed.hostname or "")
    with requests.get(url, timeout=IMAGE_URL_TIMEOUT, stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(64 * 1024):
            data.extend(chunk)
            if len(data) > IMAGE_MAX_BYTES:
                raise ValueError(f"Image exceeds IMAGE_MAX_BYTES={IMAGE_MAX_BYTES}")
    return bytes(data)


def load_image_reference(value):
    """REST 输入：data URI、纯 base64 字符串，或（开启 ALLOW_IMAGE_URLS 时）http(s) URL。"""
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return decode_image_bytes(_fetch_url(value))
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image must be a base64 string, a data URI or an http(s) URL") from exc
    return decode_image_bytes(data)
