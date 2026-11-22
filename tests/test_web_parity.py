"""Веб-версия (docs/index.html) и Python дают одинаковые цифры на всех готовых сценариях.

Тест запускает расчётное ядро страницы через Node.js; если Node.js не установлен, тест пропускается.
"""
import json
import shutil
import subprocess

import pytest

from sizing.catalog import ROOT, load_gpus, load_models, load_scenario
from sizing.cost import estimate
from sizing.plan import best_plan

NODE = shutil.which("node")
SCENARIOS = sorted((ROOT / "config" / "scenarios").glob("*.yaml"))


@pytest.mark.skipif(NODE is None, reason="Node.js не установлен")
@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_web_matches_python(path):
    sc = load_scenario(path)
    payload = {
        "model": sc.model, "quant": sc.quant, "kv": sc.kv_dtype, "users": sc.users,
        "rpd": sc.requests_per_user_per_day, "calls": sc.llm_calls_per_request, "peak": sc.peak_hour_share,
        "prompt": sc.avg_prompt_tokens, "output": sc.avg_output_tokens, "ttft": sc.target_ttft_s,
        "tpot": sc.target_tpot_ms, "ha": sc.ha, "always": sc.always_on, "days": sc.working_days_per_month,
        "apiIn": sc.api_usd_per_m_input, "apiOut": sc.api_usd_per_m_output,
    }
    res = subprocess.run([NODE, str(ROOT / "tests" / "web_engine.js")], input=json.dumps(payload),
                         capture_output=True, text=True, check=True, encoding="utf-8")
    web = json.loads(res.stdout)
    models = load_models()
    gpus, econ = load_gpus()
    for name, g in gpus.items():
        p = best_plan(sc, models[sc.model], g)
        w = web[name]
        if not p.feasible:
            assert w is None, f"{name}: Python не нашёл варианта, а веб-версия нашла"
            continue
        c = estimate(sc, g, p, econ)
        assert (w["tp"], w["batch"], w["replicas"], w["gpus"]) == (p.tp, p.batch, p.replicas, p.total_gpus), name
        assert w["ttft"] == pytest.approx(p.ttft_s, rel=1e-9), name
        assert w["tpot_ms"] == pytest.approx(p.tpot_ms, rel=1e-9), name
        for key, val in (("own", c.onprem_total), ("cloud", c.cloud_total), ("api", c.api_total)):
            assert w[key] == pytest.approx(val, rel=1e-9), f"{name}: {key}"
