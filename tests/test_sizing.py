"""Проверки формул и логики подбора."""
from dataclasses import replace

import pytest

from sizing.catalog import ROOT, Scenario, load_gpus, load_models, load_scenario
from sizing.cost import estimate
from sizing.memory import kv_bytes_per_token, kv_gb_per_sequence, weights_gb
from sizing.perf import moe_weight_fraction, tpot_s
from sizing.plan import best_plan, plan_for_tp

M = load_models()
G, ECON = load_gpus()


def test_kv_per_token_matches_known_value():
    # Llama 3.1 70B: 2 × 80 слоёв × 8 KV-голов × 128 × 2 байта = 327 680 байт (320 КБ)
    assert kv_bytes_per_token(M["llama-3.1-70b"], "fp16") == 327_680
    # 8 000 токенов на запрос: около 2,6 ГБ
    assert kv_gb_per_sequence(M["llama-3.1-70b"], "fp16", 8000) == pytest.approx(2.62, abs=0.01)


def test_gqa_matters():
    # Без grouped-query attention KV-кэш был бы в 8 раз больше (64 головы вместо 8)
    m = M["llama-3.1-70b"]
    assert kv_bytes_per_token(replace(m, kv_heads=m.attention_heads), "fp16") == 8 * kv_bytes_per_token(m, "fp16")


def test_weights_by_quantization():
    m = M["qwen2.5-72b"]
    assert weights_gb(m, "fp16") == pytest.approx(145.4, abs=0.1)
    assert weights_gb(m, "fp8") == pytest.approx(72.7, abs=0.1)
    assert weights_gb(m, "int4") == pytest.approx(38.2, abs=0.1)


def test_72b_fp16_needs_more_than_one_h100():
    sc = Scenario(name="t", description="", model="qwen2.5-72b", quant="fp16", users=100, requests_per_user_per_day=5)
    assert not plan_for_tp(sc, M["qwen2.5-72b"], G["h100-sxm"], 1).feasible
    assert best_plan(sc, M["qwen2.5-72b"], G["h100-sxm"]).tp >= 2


def test_tp_must_divide_heads():
    sc = Scenario(name="t", description="", model="qwen2.5-7b")
    p = plan_for_tp(sc, M["qwen2.5-7b"], G["h100-sxm"], 8)   # 28 голов на 8 не делятся
    assert not p.feasible and "не делится" in p.reasons[0]


def test_more_load_never_needs_fewer_gpus():
    base = load_scenario(ROOT / "config" / "scenarios" / "assistant.yaml")
    counts = [best_plan(replace(base, users=u), M[base.model], G["h100-sxm"]).total_gpus for u in (500, 2000, 8000)]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_capacity_covers_peak_with_headroom():
    sc = load_scenario(ROOT / "config" / "scenarios" / "assistant.yaml")
    p = best_plan(sc, M[sc.model], G["h100-sxm"])
    working = p.replicas - (1 if sc.ha else 0)
    assert working * p.replica_rps * sc.target_utilization >= sc.peak_rps


def test_batching_raises_throughput_until_kv_dominates():
    m, g = M["qwen2.5-32b"], G["h100-sxm"]
    t1 = tpot_s(m, g, "int4", "fp16", 1, 1, 3000)
    t32 = tpot_s(m, g, "int4", "fp16", 1, 32, 3000)
    assert t32 > t1                    # каждый шаг дольше
    assert 32 / t32 > 5 * (1 / t1)     # но токенов в секунду в разы больше


def test_moe_reads_more_experts_as_batch_grows():
    m = M["qwen3-30b-a3b"]
    assert moe_weight_fraction(m, 1) < moe_weight_fraction(m, 16) < moe_weight_fraction(m, 256) <= 1.0
    assert moe_weight_fraction(M["qwen2.5-32b"], 1) == 1.0
    # при малом батче MoE 30B генерирует быстрее плотной 32B
    g = G["h100-sxm"]
    assert tpot_s(m, g, "fp16", "fp16", 1, 1, 2000) < tpot_s(M["qwen2.5-32b"], g, "fp16", "fp16", 1, 1, 2000)


def test_impossible_target_is_reported():
    sc = Scenario(name="t", description="", model="qwen2.5-72b", quant="fp16", target_tpot_ms=1)
    p = best_plan(sc, M["qwen2.5-72b"], G["h100-sxm"])
    assert not p.feasible and p.reasons


def test_api_cost_scales_with_volume():
    sc = load_scenario(ROOT / "config" / "scenarios" / "assistant.yaml")
    g = G["h100-sxm"]
    p = best_plan(sc, M[sc.model], g)
    c1 = estimate(sc, g, p, ECON)
    c2 = estimate(replace(sc, users=sc.users * 2), g, p, ECON)
    assert c2.api_total == pytest.approx(2 * c1.api_total)


def test_consumer_gpu_flagged_not_for_production():
    assert G["rtx-4090"].production is False
    assert all(g.production for n, g in G.items() if n != "rtx-4090")


def test_all_scenarios_load_and_have_a_production_option():
    for path in (ROOT / "config" / "scenarios").glob("*.yaml"):
        sc = load_scenario(path)
        plans = [best_plan(sc, M[sc.model], g) for g in G.values() if g.production]
        assert any(p.feasible for p in plans), path.name


def test_interactive_mode_uses_answers_and_saves_scenario(monkeypatch, tmp_path, capsys):
    from sizing import interactive
    answers = iter(["qwen2.5-14b", "fp8", "", "500", "", "", "", "1500", "", "", "", "нет", "нет",
                    "h100-sxm", str(tmp_path / "my.yaml")])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert interactive.run() == 0
    out = capsys.readouterr().out
    assert "qwen2.5-14b" in out and "резерв N+1: нет" in out
    saved = load_scenario(tmp_path / "my.yaml")
    assert saved.users == 500 and saved.avg_prompt_tokens == 1500 and saved.ha is False and saved.always_on is False


def test_invalid_answer_is_asked_again(monkeypatch):
    from sizing.interactive import ask
    answers = iter(["-5", "abc", "300"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert ask("Пользователей", 100, int, lambda v: v > 0) == 300


def test_calc_rejects_bad_values():
    from sizing.cli import main
    assert main(["calc", "--users", "0"]) == 2
    assert main(["calc", "--peak-share", "1.5"]) == 2


def test_prefill_has_weight_read_floor_for_moe_short_prompt():
    from sizing.perf import prefill_s
    from sizing.memory import weights_gb, GB
    models = load_models(); gpus, _ = load_gpus()
    m, g = models["qwen3-30b-a3b"], gpus["h100-sxm"]
    short = prefill_s(m, g, "fp16", 1, 200)
    floor = weights_gb(m, "fp16") * GB / (g.bandwidth_tbs * 1e12 * 0.6)
    assert short >= floor * 0.999          # короткий промпт: упирается в чтение всех весов
    long = prefill_s(m, g, "fp16", 1, 12000)
    assert long > floor                    # длинный: упирается в вычисления по активным параметрам
