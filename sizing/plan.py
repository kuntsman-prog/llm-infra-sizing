"""Подбор конфигурации: tensor parallel, батч, число реплик, число GPU.

Для каждого допустимого уровня tensor parallel (1, 2, 4, 8 GPU на реплику):
1. Веса должны поместиться, и должно остаться место под KV-кэш.
2. Максимальный батч ограничен памятью под KV-кэш и целевым временем на токен.
3. Пропускная способность реплики с учётом пауз на prefill:
   за время T реплика обработает n запросов, где n·prefill + (n·output/b)·TPOT = T,
   откуда μ = b / (output·TPOT + b·prefill) запросов в секунду.
4. Число реплик: пиковый поток / (μ × целевая загрузка), плюс резервная при HA.
Выбирается вариант с наименьшим числом GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import GPU, Model, Scenario
from .memory import kv_capacity_gb, kv_gb_per_sequence, weights_gb
from .perf import prefill_s, tpot_s

TP_OPTIONS = (1, 2, 4, 8)
GPUS_PER_NODE = 8


@dataclass
class Plan:
    gpu: str
    feasible: bool
    tp: int = 0
    batch: int = 0
    replicas: int = 0
    total_gpus: int = 0
    nodes: int = 0
    weights_gb: float = 0.0
    kv_per_seq_gb: float = 0.0
    kv_capacity_gb: float = 0.0
    ttft_s: float = 0.0
    tpot_ms: float = 0.0
    replica_rps: float = 0.0
    peak_rps: float = 0.0
    peak_tokens_per_s: float = 0.0
    reasons: list[str] = field(default_factory=list)


def plan_for_tp(sc: Scenario, model: Model, gpu: GPU, tp: int) -> Plan:
    p = Plan(gpu=gpu.name, feasible=False, tp=tp, peak_rps=sc.peak_rps)
    if model.attention_heads % tp:
        p.reasons.append(f"TP={tp}: число голов внимания {model.attention_heads} не делится на {tp}")
        return p
    ctx_tokens = sc.avg_prompt_tokens + sc.avg_output_tokens
    if ctx_tokens > model.max_context:
        p.reasons.append("контекст длиннее максимального для модели")
        return p
    p.weights_gb = weights_gb(model, sc.quant)
    p.kv_per_seq_gb = kv_gb_per_sequence(model, sc.kv_dtype, ctx_tokens)
    p.kv_capacity_gb = kv_capacity_gb(model, sc.quant, gpu.vram_gb, tp)
    max_batch_mem = math.floor(p.kv_capacity_gb / p.kv_per_seq_gb) if p.kv_capacity_gb > 0 else 0
    if max_batch_mem < 1:
        p.reasons.append(f"TP={tp}: веса {p.weights_gb:.0f} ГБ не оставляют места под KV-кэш")
        return p

    avg_ctx = sc.avg_prompt_tokens + sc.avg_output_tokens / 2
    target = sc.target_tpot_ms / 1000
    if tpot_s(model, gpu, sc.quant, sc.kv_dtype, tp, 1, avg_ctx) > target:
        p.reasons.append(f"TP={tp}: даже один запрос генерирует медленнее {sc.target_tpot_ms:.0f} мс на токен")
        return p
    lo, hi = 1, max_batch_mem
    while lo < hi:  # TPOT растёт с батчем: двоичный поиск наибольшего допустимого
        mid = (lo + hi + 1) // 2
        if tpot_s(model, gpu, sc.quant, sc.kv_dtype, tp, mid, avg_ctx) <= target:
            lo = mid
        else:
            hi = mid - 1
    b = lo
    tpot = tpot_s(model, gpu, sc.quant, sc.kv_dtype, tp, b, avg_ctx)
    pre = prefill_s(model, gpu, sc.quant, tp, sc.avg_prompt_tokens)
    mu = b / (sc.avg_output_tokens * tpot + b * pre)
    # Время до первого токена: свой prefill плюс ожидание prefill-ов, пришедших раньше
    rho_prefill = min(mu * sc.target_utilization * pre, 0.95)
    ttft = pre * (1 + rho_prefill / (1 - rho_prefill))
    if ttft > sc.target_ttft_s:
        p.reasons.append(f"TP={tp}: время до первого токена {ttft:.1f} с больше целевого {sc.target_ttft_s} с")
        return p

    replicas = max(1, math.ceil(sc.peak_rps / (mu * sc.target_utilization)))
    if sc.ha:
        replicas += 1
    p.feasible = True
    p.batch, p.replicas = b, replicas
    p.total_gpus = replicas * tp
    p.nodes = math.ceil(p.total_gpus / GPUS_PER_NODE)
    p.ttft_s, p.tpot_ms = ttft, tpot * 1000
    p.replica_rps = mu
    p.peak_tokens_per_s = sc.peak_rps * sc.avg_output_tokens
    return p


def best_plan(sc: Scenario, model: Model, gpu: GPU) -> Plan:
    plans = [plan_for_tp(sc, model, gpu, tp) for tp in TP_OPTIONS]
    ok = [p for p in plans if p.feasible]
    if not ok:
        bad = Plan(gpu=gpu.name, feasible=False, peak_rps=sc.peak_rps)
        bad.reasons = [r for p in plans for r in p.reasons]
        return bad
    return min(ok, key=lambda p: (p.total_gpus, p.tp))
