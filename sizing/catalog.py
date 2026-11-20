"""Каталог моделей, GPU и сценариев нагрузки."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

BYTES_PER_PARAM = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}
QUANT_OVERHEAD = {"int4": 1.05}  # масштабы и нули групповой квантизации (AWQ, GPTQ)
KV_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1}


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float
    active_params_b: float
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    max_context: int
    experts: int = 0
    experts_per_token: int = 0

    @property
    def is_moe(self) -> bool:
        return self.experts > 0


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float
    bandwidth_tbs: float
    fp16_tflops: float
    tdp_w: float
    price_usd: float
    platform_usd_per_gpu: float
    cloud_usd_per_hour: float
    interconnect: str = "pcie"
    production: bool = True
    note: str = ""


@dataclass
class Scenario:
    name: str
    description: str
    model: str
    quant: str = "fp16"
    kv_dtype: str = "fp16"
    users: int = 1000
    requests_per_user_per_day: float = 10
    llm_calls_per_request: float = 1        # агентный сценарий: несколько вызовов модели на задачу
    peak_hour_share: float = 0.2            # доля дневного трафика в пиковый час
    working_days_per_month: int = 22
    avg_prompt_tokens: int = 2000
    avg_output_tokens: int = 300
    target_ttft_s: float = 2.0              # время до первого токена
    target_tpot_ms: float = 60.0            # время на каждый следующий токен
    ha: bool = True                         # резервная реплика (N+1)
    target_utilization: float = 0.7         # запас мощности в пике
    always_on: bool = True                  # для облака: аренда круглосуточно или только в рабочие часы
    gpus: list[str] = field(default_factory=list)
    api_usd_per_m_input: float | None = None   # цена API для класса модели сценария
    api_usd_per_m_output: float | None = None

    @property
    def peak_rps(self) -> float:
        daily = self.users * self.requests_per_user_per_day * self.llm_calls_per_request
        return daily * self.peak_hour_share / 3600

    @property
    def monthly_calls(self) -> float:
        return self.users * self.requests_per_user_per_day * self.llm_calls_per_request * self.working_days_per_month


def load_models() -> dict[str, Model]:
    raw = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["models"]
    return {k: Model(name=k, **v) for k, v in raw.items()}


def load_gpus() -> tuple[dict[str, GPU], dict]:
    raw = yaml.safe_load((ROOT / "config" / "gpus.yaml").read_text(encoding="utf-8"))
    return {k: GPU(name=k, **v) for k, v in raw["gpus"].items()}, raw["economics"]


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate(Scenario(**data))


def validate(sc: Scenario) -> Scenario:
    positive = {
        "users": "число пользователей", "requests_per_user_per_day": "запросов на пользователя в день",
        "llm_calls_per_request": "вызовов модели на запрос", "avg_prompt_tokens": "длина промпта",
        "avg_output_tokens": "длина ответа", "target_ttft_s": "время до первого токена",
        "target_tpot_ms": "время на токен",
    }
    for name, label in positive.items():
        if getattr(sc, name) <= 0:
            raise ValueError(f"{label} должно быть больше нуля")
    if not 0 < sc.peak_hour_share <= 1:
        raise ValueError("доля трафика в пиковый час должна быть от 0 до 1")
    if not 0 < sc.target_utilization <= 1:
        raise ValueError("целевая загрузка должна быть от 0 до 1")
    if sc.quant not in BYTES_PER_PARAM:
        raise ValueError(f"неизвестная квантизация {sc.quant}, допустимо: {sorted(BYTES_PER_PARAM)}")
    if sc.kv_dtype not in KV_BYTES:
        raise ValueError(f"неизвестный тип KV-кэша {sc.kv_dtype}, допустимо: {sorted(KV_BYTES)}")
    return sc
