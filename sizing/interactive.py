"""Интерактивный ввод параметров: вопрос, значение по умолчанию в скобках, Enter оставляет его."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import yaml

from .catalog import ROOT, Scenario, load_gpus, load_models, validate
from .cli import report


def ask(prompt: str, default, cast: Callable = str, check: Callable | None = None, hint: str = ""):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = cast(raw.replace(",", ".")) if cast in (int, float) else cast(raw)
            if check and not check(value):
                raise ValueError
            return value
        except ValueError:
            print(f"  некорректное значение{': ' + hint if hint else ''}")


def yes(raw: str) -> bool:
    if raw.lower() in ("да", "д", "y", "yes", "1"):
        return True
    if raw.lower() in ("нет", "н", "n", "no", "0"):
        return False
    raise ValueError


def choose(prompt: str, options: list[str], default: str) -> str:
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    def pick(raw: str) -> str:
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        raise ValueError
    return ask(prompt, default, pick, hint="номер или название из списка")


def run() -> int:
    models = load_models()
    gpus, _ = load_gpus()
    d = Scenario(name="", description="", model="qwen2.5-32b", quant="int4")
    print("Калькулятор сайзинга LLM-инфраструктуры. Enter оставляет значение в скобках.\n")

    print("Модель:")
    model = choose("Выберите модель", list(models), d.model)
    print("Квантизация весов: fp16 (без потерь), fp8, int8, int4 (в 4 раза меньше памяти, возможна потеря качества)")
    quant = ask("Квантизация весов", d.quant, str, lambda v: v in ("fp16", "bf16", "fp8", "int8", "int4"), "fp16, bf16, fp8, int8 или int4")
    kv = ask("Тип KV-кэша", d.kv_dtype, str, lambda v: v in ("fp16", "bf16", "fp8"), "fp16, bf16 или fp8")

    print("\nНагрузка:")
    users = ask("Пользователей", d.users, int, lambda v: v > 0, "целое число больше нуля")
    rpd = ask("Запросов на пользователя в день", d.requests_per_user_per_day, float, lambda v: v > 0)
    calls = ask("Вызовов модели на один запрос (для агентов больше 1)", d.llm_calls_per_request, float, lambda v: v > 0)
    peak = ask("Доля дневного трафика в пиковый час, от 0 до 1", d.peak_hour_share, float, lambda v: 0 < v <= 1)
    prompt = ask("Средняя длина промпта, токенов (с найденными фрагментами для RAG)", d.avg_prompt_tokens, int, lambda v: v > 0)
    output = ask("Средняя длина ответа, токенов", d.avg_output_tokens, int, lambda v: v > 0)

    print("\nТребования:")
    ttft = ask("Время до первого токена не больше, секунд", d.target_ttft_s, float, lambda v: v > 0)
    tpot = ask("Время на каждый следующий токен не больше, мс (60 мс быстрее чтения)", d.target_tpot_ms, float, lambda v: v > 0)
    ha = ask("Нужна резервная реплика N+1, да/нет", "да", yes, hint="да или нет")
    always = ask("Облако арендуется круглосуточно, да/нет (нет: только рабочие часы)", "да", yes, hint="да или нет")

    print("\nGPU для сравнения (Enter: все):")
    raw = input(f"  через запятую из {', '.join(gpus)}: ").strip()
    only = [g.strip() for g in raw.split(",") if g.strip()] or None
    if only and any(g not in gpus for g in only):
        print("  неизвестный GPU в списке, сравниваю все")
        only = None

    sc = validate(Scenario(name="Интерактивный расчёт", description="", model=model, quant=quant, kv_dtype=kv,
                           users=users, requests_per_user_per_day=rpd, llm_calls_per_request=calls,
                           peak_hour_share=peak, avg_prompt_tokens=prompt, avg_output_tokens=output,
                           target_ttft_s=ttft, target_tpot_ms=tpot, ha=ha, always_on=always))
    print()
    report(sc, only)

    path = input("\nСохранить параметры как сценарий? Имя файла или Enter, чтобы пропустить: ").strip()
    if path:
        target = Path(path)
        if not target.suffix:
            target = ROOT / "config" / "scenarios" / f"{path}.yaml"
        data = {k: v for k, v in asdict(sc).items() if v is not None and v != []}
        target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Сохранено: {target}. Повторить расчёт: python -m sizing.cli scenario {target}")
    return 0
