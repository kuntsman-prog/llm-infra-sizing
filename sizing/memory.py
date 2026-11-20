"""Память GPU: веса модели и KV-кэш.

Веса: число параметров × байт на параметр (зависит от квантизации).
KV-кэш: для каждого токена в контексте каждый слой хранит ключи и значения
по всем KV-головам: 2 × слои × KV-головы × размер головы × байт на значение.
Grouped-query attention (KV-голов меньше, чем голов внимания) сокращает KV-кэш
в разы, поэтому считать надо по kv_heads, а не по attention_heads.
"""
from __future__ import annotations

from .catalog import BYTES_PER_PARAM, KV_BYTES, QUANT_OVERHEAD, Model

GB = 1e9
VRAM_UTILIZATION = 0.90     # доля памяти, которую отдаёт движку инференса (как gpu_memory_utilization в vLLM)
RUNTIME_RESERVE_GB = 1.5    # на каждый GPU: CUDA-контекст, активации, буферы


def weights_gb(model: Model, quant: str) -> float:
    return model.params_b * BYTES_PER_PARAM[quant] * QUANT_OVERHEAD.get(quant, 1.0)


def kv_bytes_per_token(model: Model, kv_dtype: str) -> int:
    return 2 * model.layers * model.kv_heads * model.head_dim * KV_BYTES[kv_dtype]


def kv_gb_per_sequence(model: Model, kv_dtype: str, tokens: int) -> float:
    return kv_bytes_per_token(model, kv_dtype) * tokens / GB


def kv_capacity_gb(model: Model, quant: str, vram_gb: float, tp: int) -> float:
    """Память под KV-кэш на одну реплику из tp GPU после размещения весов."""
    return tp * (vram_gb * VRAM_UTILIZATION - RUNTIME_RESERVE_GB) - weights_gb(model, quant)
