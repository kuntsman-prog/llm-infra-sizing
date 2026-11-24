"""Веб-версия (docs/index.html) и Python дают одинаковые цифры на всех готовых сценариях.

Тест запускает расчётное ядро страницы через Node.js; если Node.js не установлен, тест пропускается.
"""
import json
import shutil
import subprocess

import pytest

from dataclasses import replace

from sizing.catalog import ROOT, load_gpus, load_models, load_scenario
from sizing.cost import estimate
from sizing.plan import best_plan

NODE = shutil.which("node")
SCENARIOS = sorted((ROOT / "config" / "scenarios").glob("*.yaml"))


# Готовые сценарии и их варианты с новыми параметрами: кэш префиксов, KV-кэш вручную,
# FP8/FP4, резерв в облаке, доля инфраструктуры, цена кэшированных токенов API
VARIANTS = {
    "as-is": ({}, {}),
    "prefix-reserved-infra": ({"prefix_cache_hit": 0.8, "api_usd_per_m_cached": 0.1},
                              {"cloud_reserved": True, "reserved_discount": 0.4, "infra_share": 0.15}),
    "fp4-kv-manual": ({"quant": "fp4", "kv_kb_per_token": 96.0}, {}),
}
CASES = [(p, v) for p in SCENARIOS for v in VARIANTS] + [("deepseek", "as-is"), ("deepseek", "prefix-reserved-infra")]


def _scenario(path):
    if path == "deepseek":
        sc = load_scenario(ROOT / "config" / "scenarios" / "agents.yaml")
        return replace(sc, model="deepseek-v3", quant="fp8", kv_dtype="fp8")
    return load_scenario(path)


@pytest.mark.skipif(NODE is None, reason="Node.js не установлен")
@pytest.mark.parametrize("path,variant", CASES, ids=lambda x: x.stem if hasattr(x, "stem") else str(x))
def test_web_matches_python(path, variant):
    sc_over, econ_over = VARIANTS[variant]
    sc = replace(_scenario(path), **sc_over)
    payload = {
        "model": sc.model, "quant": sc.quant, "kv": sc.kv_dtype, "users": sc.users,
        "rpd": sc.requests_per_user_per_day, "calls": sc.llm_calls_per_request, "peak": sc.peak_hour_share,
        "prompt": sc.avg_prompt_tokens, "output": sc.avg_output_tokens, "ttft": sc.target_ttft_s,
        "tpot": sc.target_tpot_ms, "ha": sc.ha, "always": sc.always_on, "days": sc.working_days_per_month,
        "apiIn": sc.api_usd_per_m_input, "apiOut": sc.api_usd_per_m_output, "apiCached": sc.api_usd_per_m_cached,
        "prefixHit": sc.prefix_cache_hit, "kvKb": sc.kv_kb_per_token,
        "reserved": econ_over.get("cloud_reserved", False), "reservedDiscount": econ_over.get("reserved_discount"),
        "infra": econ_over.get("infra_share", 0.0),
    }
    res = subprocess.run([NODE, str(ROOT / "tests" / "web_engine.js")], input=json.dumps(payload),
                         capture_output=True, text=True, check=True, encoding="utf-8")
    web = json.loads(res.stdout)
    models = load_models()
    gpus, econ = load_gpus()
    econ.update(econ_over)
    assert set(web) == set(gpus), "списки GPU в веб-версии и в config/gpus.yaml различаются"
    for name, g in gpus.items():
        p = best_plan(sc, models[sc.model], g)
        w = web[name]
        if not p.feasible:
            assert w is None, f"{name}: Python не нашёл варианта, а веб-версия нашла"
            continue
        assert w is not None, f"{name}: веб-версия не нашла варианта, а Python нашёл"
        c = estimate(sc, g, p, econ)
        assert (w["tp"], w["batch"], w["replicas"], w["gpus"]) == (p.tp, p.batch, p.replicas, p.total_gpus), name
        assert w["ttft"] == pytest.approx(p.ttft_s, rel=1e-9), name
        assert w["tpot_ms"] == pytest.approx(p.tpot_ms, rel=1e-9), name
        for key, val in (("own", c.onprem_total), ("cloud", c.cloud_total), ("api", c.api_total)):
            assert w[key] == pytest.approx(val, rel=1e-9), f"{name}: {key}"
