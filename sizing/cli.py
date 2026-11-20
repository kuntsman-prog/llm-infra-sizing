"""Командная строка.

  python -m sizing.cli ask                                          ввод параметров по вопросам
  python -m sizing.cli calc --model qwen2.5-32b --users 2000 ...    расчёт с параметрами в командной строке
  python -m sizing.cli scenario config/scenarios/assistant.yaml   расчёт по готовому сценарию
  python -m sizing.cli kv qwen2.5-72b --context 8000 --sequences 32 KV-кэш для заданной нагрузки
  python -m sizing.cli models                                       каталог моделей
  python -m sizing.cli gpus                                         каталог GPU
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog import Scenario, load_gpus, load_models, load_scenario, validate
from .cost import estimate
from .memory import kv_bytes_per_token, kv_gb_per_sequence, weights_gb
from .plan import best_plan


def money(x: float) -> str:
    if x >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x / 1e3:.0f}K"


def cmd_scenario(path: Path, only: list[str] | None) -> int:
    return report(load_scenario(path), only)


def report(sc, only: list[str] | None = None) -> int:
    models = load_models()
    gpus, econ = load_gpus()
    if sc.model not in models:
        print(f"модель {sc.model} не найдена в config/models.yaml")
        return 1
    m = models[sc.model]
    names = only or sc.gpus or list(gpus)

    print(f"Сценарий: {sc.name}")
    print(f"Модель: {m.name} ({m.params_b} млрд параметров, активных {m.active_params_b}), веса {sc.quant}, KV-кэш {sc.kv_dtype}")
    print(f"Пиковый поток: {sc.peak_rps:.2f} вызовов модели в секунду; в месяц {sc.monthly_calls:,.0f} вызовов".replace(",", " "))
    ctx = sc.avg_prompt_tokens + sc.avg_output_tokens
    print(f"Память: веса {weights_gb(m, sc.quant):.1f} ГБ; KV-кэш {kv_bytes_per_token(m, sc.kv_dtype) / 1024:.0f} КБ на токен, "
          f"{kv_gb_per_sequence(m, sc.kv_dtype, ctx):.2f} ГБ на запрос в {ctx} токенов")
    print(f"Цели: время до первого токена до {sc.target_ttft_s} с, на токен до {sc.target_tpot_ms:.0f} мс; "
          f"резерв N+1: {'да' if sc.ha else 'нет'}\n")

    header = f"{'GPU':<11}{'TP':>3}{'реплик':>8}{'GPU':>5}{'батч':>6}{'TTFT, с':>9}{'мс/ток':>8}   {'своё 3 г':>9}{'облако 3 г':>12}{'API 3 г':>9}   {'$/1М ток. своё':>15}"
    print(header)
    print("-" * len(header))
    feasible = []
    small_batch = False
    for name in names:
        g = gpus[name]
        p = best_plan(sc, m, g)
        if not p.feasible:
            print(f"{name:<11}  не подходит: {p.reasons[-1] if p.reasons else ''}")
            continue
        c = estimate(sc, g, p, econ)
        if g.production:
            feasible.append((p, c))
        mark = "" if g.production else "  *стенд"
        if p.batch < 4:
            mark += "  !малый батч"
            small_batch = True
        print(f"{name:<11}{p.tp:>3}{p.replicas:>8}{p.total_gpus:>5}{p.batch:>6}{p.ttft_s:>9.2f}{p.tpot_ms:>8.0f}   "
              f"{money(c.onprem_total):>9}{money(c.cloud_total):>12}{money(c.api_total):>9}   {c.per_m_tokens(c.onprem_total):>15.3f}{mark}")

    if any(not gpus[n].production for n in names):
        print("* только для стендов и пилотов, в рекомендации не участвует")
    if small_batch:
        print("! малый батч: GPU едва укладывается в цель по скорости и используется неэффективно")
    if not feasible:
        print("\nНи один GPU не обеспечивает целевые показатели: смягчите цели, выберите меньшую модель или квантизацию.")
        return 0
    p, c = min(feasible, key=lambda pc: pc[1].onprem_total)
    best_cloud = min(feasible, key=lambda pc: pc[1].cloud_total)
    print(f"\nДешевле всего своё оборудование на {p.gpu}: {p.total_gpus} GPU ({p.replicas} реплик по {p.tp}), "
          f"{p.nodes} сервер(а), {money(c.onprem_total)} за {econ['years']} года.")
    print(f"Аренда в облаке: от {money(best_cloud[1].cloud_total)} ({best_cloud[0].gpu}); API провайдера: {money(c.api_total)}.")
    cheapest = min(("своё оборудование", c.onprem_total), ("облако", best_cloud[1].cloud_total), ("API", c.api_total), key=lambda x: x[1])
    print(f"По цене на горизонте {econ['years']} лет выигрывает: {cheapest[0]}. Учтите требования к данным: "
          f"API подходит, только если данные можно передавать провайдеру.")
    print("Перед закупкой: нагрузочный тест на vLLM с реальными промптами; оценка имеет точность ±30–50%.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sizing", description="Сайзинг LLM-инфраструктуры: GPU, задержка, стоимость.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenario")
    s.add_argument("path", type=Path)
    s.add_argument("--gpu", action="append")
    sub.add_parser("ask", help="ввод параметров по вопросам")
    c = sub.add_parser("calc", help="расчёт с параметрами в командной строке", add_help=False,
                       description="Расчёт конфигурации GPU и стоимости. Не указанные параметры берутся по умолчанию.")
    c.add_argument("-h", "--help", action="help", help="показать эту справку")
    d = Scenario(name="", description="", model="qwen2.5-32b")
    c.add_argument("--model", default=d.model, help=f"модель из config/models.yaml (по умолчанию {d.model}); список: python -m sizing.cli models")
    c.add_argument("--quant", default=d.quant, choices=["fp16", "bf16", "fp8", "int8", "int4"], help="квантизация весов: int4 в 4 раза меньше памяти, чем fp16")
    c.add_argument("--kv-dtype", default=d.kv_dtype, choices=["fp16", "bf16", "fp8"], help="тип KV-кэша: fp8 вдвое меньше памяти под контекст")
    c.add_argument("--users", type=int, default=d.users, help=f"число пользователей (по умолчанию {d.users})")
    c.add_argument("--requests-per-day", type=float, default=d.requests_per_user_per_day, help="запросов на пользователя в день")
    c.add_argument("--calls-per-request", type=float, default=d.llm_calls_per_request, help="вызовов модели на запрос (агенты: больше 1)")
    c.add_argument("--peak-share", type=float, default=d.peak_hour_share, help="доля дневного трафика в пиковый час")
    c.add_argument("--prompt", type=int, default=d.avg_prompt_tokens, help="средняя длина промпта, токенов")
    c.add_argument("--output", type=int, default=d.avg_output_tokens, help="средняя длина ответа, токенов")
    c.add_argument("--ttft", type=float, default=d.target_ttft_s, help="целевое время до первого токена, с")
    c.add_argument("--tpot", type=float, default=d.target_tpot_ms, help="целевое время на токен, мс")
    c.add_argument("--no-ha", action="store_true", help="без резервной реплики")
    c.add_argument("--working-hours", action="store_true", help="облако арендуется только в рабочие часы")
    c.add_argument("--gpu", action="append", help="ограничить список GPU; можно указать несколько раз")
    k = sub.add_parser("kv")
    k.add_argument("model")
    k.add_argument("--context", type=int, required=True)
    k.add_argument("--sequences", type=int, default=1)
    k.add_argument("--kv-dtype", default="fp16")
    sub.add_parser("models")
    sub.add_parser("gpus")
    a = p.parse_args(argv)

    if a.cmd == "scenario":
        return cmd_scenario(a.path, a.gpu)
    if a.cmd == "ask":
        from .interactive import run
        return run()
    if a.cmd == "calc":
        try:
            sc = validate(Scenario(
                name="Расчёт из командной строки", description="", model=a.model, quant=a.quant, kv_dtype=a.kv_dtype,
                users=a.users, requests_per_user_per_day=a.requests_per_day, llm_calls_per_request=a.calls_per_request,
                peak_hour_share=a.peak_share, avg_prompt_tokens=a.prompt, avg_output_tokens=a.output,
                target_ttft_s=a.ttft, target_tpot_ms=a.tpot, ha=not a.no_ha, always_on=not a.working_hours))
        except ValueError as exc:
            print(f"ошибка в параметрах: {exc}")
            return 2
        return report(sc, a.gpu)
    if a.cmd == "kv":
        m = load_models()[a.model]
        per_tok = kv_bytes_per_token(m, a.kv_dtype)
        total = kv_gb_per_sequence(m, a.kv_dtype, a.context) * a.sequences
        print(f"{m.name}: {per_tok / 1024:.0f} КБ на токен; {a.sequences} запрос(ов) по {a.context} токенов: {total:.1f} ГБ KV-кэша")
        for q in ("fp16", "fp8", "int4"):
            print(f"  веса {q}: {weights_gb(m, q):.1f} ГБ")
    elif a.cmd == "models":
        for m in load_models().values():
            moe = f", MoE {m.experts_per_token} из {m.experts} экспертов" if m.is_moe else ""
            print(f"{m.name:<15} {m.params_b:>5} млрд, слоёв {m.layers}, KV-голов {m.kv_heads}{moe}")
    elif a.cmd == "gpus":
        gpus, _ = load_gpus()
        for g in gpus.values():
            print(f"{g.name:<11} {g.vram_gb:>4.0f} ГБ, {g.bandwidth_tbs:.2f} ТБ/с, {g.fp16_tflops:.0f} TFLOPS FP16, {g.tdp_w:.0f} Вт  {g.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
