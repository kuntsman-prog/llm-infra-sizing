"""Контрольный пример: ассистент с RAG на H100, посчитанный вручную по шагам.

Каждое число ниже можно получить на калькуляторе или пересчитать на бумаге,
формулы приведены в комментариях. Тот же пример разобран в GUIDE.ru.md
(раздел «Контрольный пример») и используется для демонстрации.

Входные данные (config/scenarios/assistant.yaml):
  qwen2.5-32b: 32,5 млрд параметров, 64 слоя, 8 KV-голов, размер головы 128;
  веса int4, KV-кэш fp16; 2 000 пользователей × 15 запросов, 1 вызов, пик 20%;
  промпт 3 000 токенов, ответ 350; цели: TTFT до 2 с, до 60 мс на токен; N+1.
  H100 SXM: 80 ГБ, 3,35 ТБ/с, 989 TFLOPS fp16, NVLink, $30 000 + $8 000 сервер, $3/ч.
"""
import pytest

from sizing.catalog import ROOT, load_gpus, load_models, load_scenario
from sizing.cost import estimate
from sizing.memory import kv_bytes_per_token, kv_capacity_gb, kv_gb_per_sequence, weights_gb
from sizing.perf import prefill_s, tpot_s
from sizing.plan import best_plan, plan_for_tp

SC = load_scenario(ROOT / "config" / "scenarios" / "assistant.yaml")
M = load_models()[SC.model]
GPUS, ECON = load_gpus()
H100 = GPUS["h100-sxm"]


def test_step1_load():
    # 2 000 × 15 × 1 = 30 000 вызовов в день; 20% в пиковый час: 6 000 / 3 600 с = 1,67 вызова/с
    assert SC.peak_rps == pytest.approx(1.667, abs=0.001)
    # 30 000 × 22 рабочих дня = 660 000 вызовов в месяц
    assert SC.monthly_calls == 660_000


def test_step2_memory():
    # Веса: 32,5 млрд × 0,5 байта × 1,05 (масштабы квантизации int4) = 17,06 ГБ
    assert weights_gb(M, "int4") == pytest.approx(17.06, abs=0.01)
    # KV на токен: 2 (ключ и значение) × 64 слоя × 8 KV-голов × 128 × 2 байта = 262 144 байта (256 КБ)
    assert kv_bytes_per_token(M, "fp16") == 262_144
    # KV на запрос: 262 144 × (3 000 + 350) токенов = 0,878 ГБ
    assert kv_gb_per_sequence(M, "fp16", 3350) == pytest.approx(0.878, abs=0.001)
    # Свободно под KV на одной H100: 80 × 0,9 − 1,5 (служебное) − 17,06 (веса) = 53,44 ГБ
    assert kv_capacity_gb(M, "int4", H100.vram_gb, 1) == pytest.approx(53.44, abs=0.01)
    # Поместится запросов: 53,44 / 0,878 = 60,8, то есть 60


def test_step3_speed():
    # Батч упирается в память (60), а не в скорость: при 60 запросах шаг генерации
    # читает веса и KV всех запросов: 17,06 + 60 × 3 175 (средний контекст) × 262 144 байт = 67,0 ГБ;
    # 67,0 ГБ / (3,35 ТБ/с × 0,6) = 33 мс на токен, это меньше цели 60 мс
    avg_ctx = 3000 + 350 / 2
    assert tpot_s(M, H100, "int4", "fp16", 1, 60, avg_ctx) * 1000 == pytest.approx(33.3, abs=0.1)
    # Промпт: 2 × 32,5 млрд × 3 000 токенов / (989 TFLOPS × 0,4) = 0,49 с
    assert prefill_s(M, H100, "int4", 1, 3000) == pytest.approx(0.493, abs=0.001)


def test_step4_replicas_and_ttft():
    p = plan_for_tp(SC, M, H100, 1)
    assert p.feasible and p.batch == 60
    # Пропускная способность копии модели: 60 / (350 × 0,0333 + 60 × 0,493) = 1,45 запроса/с
    assert p.replica_rps == pytest.approx(1.455, abs=0.001)
    # Копий: 1,67 / (1,45 × 0,7) = 1,64, округляем до 2, плюс резервная: 3
    assert p.replicas == 3 and p.total_gpus == 3
    # Очередь на обработку промптов: загрузка 1,45 × 0,7 × 0,49 = 0,50;
    # TTFT = 0,49 × (1 + 0,50 / 0,50) = 0,99 с, это меньше цели 2 с
    assert p.ttft_s == pytest.approx(0.99, abs=0.01)
    # TP=1 даёт меньше всего карт, поэтому калькулятор выбирает именно его
    assert best_plan(SC, M, H100).total_gpus == 3


def test_step5_cost_three_years():
    p = plan_for_tp(SC, M, H100, 1)
    c = estimate(SC, H100, p, ECON)
    # Оборудование: 3 × ($30 000 + $8 000) = $114 000
    assert c.onprem_capex == 114_000
    # Электричество: 3 × 0,7 кВт × 1,6 (сервер) × 1,5 (PUE) = 5,04 кВт × 8 760 ч × 3 года × $0,10 = $13 245
    # Обслуживание: $114 000 × 10% × 3 года = $34 200; итого своё $161 445
    assert c.onprem_total == pytest.approx(161_445, abs=1)
    # Облако: 3 GPU × $3 × 730 ч × 36 месяцев = $236 520
    assert c.cloud_total == pytest.approx(236_520, abs=1)
    # API: 660 000 × 36 = 23,76 млн вызовов × (3 000 × $0,50 + 350 × $1,50) / 1 млн = $48 114
    assert c.api_total == pytest.approx(48_114, abs=1)
