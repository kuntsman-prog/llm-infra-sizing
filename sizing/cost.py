"""Стоимость владения: своё оборудование, аренда GPU в облаке, API провайдера.

Не учтены: зарплаты команды эксплуатации, лицензии, налоги.
Для своего оборудования добавлено обслуживание как доля капитальных затрат в год, а сеть между
серверами, СХД и размещение в стойках можно учесть долей infra_share к оборудованию.
Облако: почасовая оплата по требованию или резервирование на весь срок со скидкой (cloud_reserved).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import GPU, Scenario
from .plan import Plan

HOURS_PER_YEAR = 24 * 365


@dataclass
class Cost:
    onprem_capex: float
    onprem_opex_total: float
    onprem_total: float
    cloud_total: float
    api_total: float
    tokens_total_m: float

    def per_m_tokens(self, total: float) -> float:
        return total / self.tokens_total_m if self.tokens_total_m else math.inf


def estimate(sc: Scenario, gpu: GPU, plan: Plan, econ: dict) -> Cost:
    years = econ["years"]
    months = 12 * years
    capex = plan.total_gpus * (gpu.price_usd + gpu.platform_usd_per_gpu) * (1 + econ.get("infra_share", 0.0))
    power_kw = plan.total_gpus * gpu.tdp_w / 1000 * econ["server_power_overhead"] * econ["pue"]
    energy = power_kw * HOURS_PER_YEAR * years * econ["electricity_usd_per_kwh"]
    maintenance = capex * econ["maintenance_share_per_year"] * years
    if econ.get("cloud_reserved"):   # резерв оплачивается круглосуточно, но со скидкой
        hours, rate = HOURS_PER_YEAR / 12, gpu.cloud_usd_per_hour * (1 - econ.get("reserved_discount", 0.0))
    else:
        hours = HOURS_PER_YEAR / 12 if sc.always_on else sc.working_days_per_month * 12
        rate = gpu.cloud_usd_per_hour
    cloud = plan.total_gpus * rate * hours * months
    calls = sc.monthly_calls * months
    price_in = sc.api_usd_per_m_input if sc.api_usd_per_m_input is not None else econ["api_usd_per_m_input"]
    price_out = sc.api_usd_per_m_output if sc.api_usd_per_m_output is not None else econ["api_usd_per_m_output"]
    price_cached = sc.api_usd_per_m_cached if sc.api_usd_per_m_cached is not None else price_in
    hit = sc.prefix_cache_hit
    api = calls * (sc.avg_prompt_tokens * ((1 - hit) * price_in + hit * price_cached) + sc.avg_output_tokens * price_out) / 1e6
    tokens_m = calls * (sc.avg_prompt_tokens + sc.avg_output_tokens) / 1e6
    return Cost(capex, energy + maintenance, capex + energy + maintenance, cloud, api, tokens_m)
