"""Оценка производительности инференса по roofline-модели.

Генерация (decode) ограничена пропускной способностью памяти: на каждом шаге
GPU читает веса модели и KV-кэш всех запросов в батче, а выдаёт по одному
токену на запрос. Поэтому батчинг почти бесплатно увеличивает пропускную
способность, пока чтение KV-кэша не становится сравнимым с чтением весов.

Обработка промпта (prefill) ограничена вычислениями: около 2 × активные
параметры FLOP на токен.

Это оценка порядка величины (точность ±30–50%), а не замена нагрузочного теста.
Коэффициенты эффективности отражают типичную долю паспортной мощности,
которую достигают движки вроде vLLM.
"""
from __future__ import annotations

from .catalog import BYTES_PER_PARAM, KV_BYTES, GPU, Model
from .memory import GB, kv_bytes_per_token, weights_gb

MEMORY_EFFICIENCY = 0.6
COMPUTE_EFFICIENCY = 0.4
# Потери на обмен между GPU в tensor parallel на каждый дополнительный GPU:
# с NVLink обмен быстрый, через PCIe заметно медленнее
TP_COMM_OVERHEAD = {"nvlink": 0.04, "pcie": 0.12}


def moe_weight_fraction(model: Model, batch: int) -> float:
    """Доля весов, читаемых за шаг генерации.

    Плотная модель читает все веса. MoE читает активных экспертов, но чем больше
    батч, тем больше разных экспертов задействовано: вероятность, что эксперт
    не нужен ни одному из b токенов, равна (1 - k/E)^b.
    """
    if not model.is_moe:
        return 1.0
    active = model.active_params_b / model.params_b
    coverage = 1 - (1 - model.experts_per_token / model.experts) ** batch
    return active + (1 - active) * coverage


def compute_tflops(gpu: GPU, quant: str) -> float:
    """Производительность тензорных ядер для формата весов.

    FP8 и FP4 (NVFP4) считаются на тензорных ядрах своего формата, если GPU их поддерживает.
    INT4 (AWQ, GPTQ) и INT8 рассматриваются как квантизация только весов: вычисления идут в FP16.
    """
    if quant == "fp8" and gpu.fp8_tflops:
        return gpu.fp8_tflops
    if quant == "fp4" and gpu.fp4_tflops:
        return gpu.fp4_tflops
    return gpu.fp16_tflops


def _comm(gpu: GPU, tp: int) -> float:
    return 1 + TP_COMM_OVERHEAD[gpu.interconnect] * (tp - 1)


def tpot_s(model: Model, gpu: GPU, quant: str, kv_dtype: str, tp: int, batch: int, avg_context: float) -> float:
    """Время одного шага генерации для батча из batch запросов (секунды на токен для каждого запроса)."""
    bytes_read = (weights_gb(model, quant) * GB * moe_weight_fraction(model, batch)
                  + batch * avg_context * kv_bytes_per_token(model, kv_dtype))
    t_mem = bytes_read / (tp * gpu.bandwidth_tbs * 1e12 * MEMORY_EFFICIENCY)
    t_cmp = 2 * model.active_params_b * 1e9 * batch / (tp * compute_tflops(gpu, quant) * 1e12 * COMPUTE_EFFICIENCY)
    return max(t_mem, t_cmp) * _comm(gpu, tp)


def prefill_s(model: Model, gpu: GPU, quant: str, tp: int, prompt_tokens: float) -> float:
    """Время обработки промпта одного запроса.

    Вычисления: около 2 × активные параметры FLOP на токен. Для MoE это верно и при
    длинном промпте: разные токены идут через разных экспертов, но каждый токен
    проходит только через своих, поэтому объём вычислений определяют активные параметры.
    Нижняя граница: за prefill веса читаются из памяти хотя бы один раз (для MoE
    на длинном промпте задействованы почти все эксперты); на коротких промптах
    эта граница становится определяющей.
    """
    t_cmp = 2 * model.active_params_b * 1e9 * prompt_tokens / (tp * compute_tflops(gpu, quant) * 1e12 * COMPUTE_EFFICIENCY)
    t_mem = weights_gb(model, quant) * GB / (tp * gpu.bandwidth_tbs * 1e12 * MEMORY_EFFICIENCY)
    return max(t_cmp, t_mem) * _comm(gpu, tp)


__all__ = ["tpot_s", "prefill_s", "compute_tflops", "moe_weight_fraction", "BYTES_PER_PARAM", "KV_BYTES"]
