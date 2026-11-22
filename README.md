# LLM infrastructure sizing and a reference AI platform architecture

**English** | [Русский](README.ru.md)

Two tools for the first stage of any AI platform, when you need to answer how many GPUs are needed, which ones, and what it will cost:

1. **Sizing calculator.** Given a model, quantization, load profile and target latency, it picks a configuration for different GPUs: tensor parallelism, batch size, number of replicas and servers. It compares the three-year cost of owned hardware, rented cloud GPUs and a provider API.
2. **Reference architecture of an AI platform built from scratch:** [AI_PLATFORM_ARCHITECTURE.md](AI_PLATFORM_ARCHITECTURE.md). What to build and what not to build yet, model strategy, LLM platform, knowledge and RAG, agents and their permissions, security and DR, cloud vs on-premise, phases, hiring order, risks.

Console output and scenario descriptions are in Russian.

## Web version

`docs/index.html` is the same calculator as a single page that runs in the browser: pick a preset scenario or enter your own load, and the GPU table, latency and three-year cost recalculate on every change. GPU prices and engineering assumptions (PUE, maintenance, discount rate, engine efficiency, target utilization) are editable on the page, and the table can be reduced to business columns. The page is in Russian and needs no server.

To publish it on GitHub Pages: Settings → Pages → Build and deployment → Deploy from a branch → `main`, folder `/docs`. A user guide for business stakeholders explaining every field with examples (in Russian): [GUIDE.ru.md](GUIDE.ru.md). The page will be available at `https://kuntsman-prog.github.io/llm-infra-sizing/`. It reproduces the Python calculation exactly for the three bundled scenarios.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m sizing.cli ask                                         # enter parameters step by step
python -m sizing.cli calc --model qwen2.5-14b --quant fp8 --users 500 --prompt 1500 --gpu l40s --gpu h100-sxm
python -m sizing.cli calc --help                                 # all command-line parameters
python -m sizing.cli scenario config/scenarios/assistant.yaml   # ready-made scenario: enterprise RAG assistant
python -m sizing.cli scenario config/scenarios/agents.yaml      # agent platform on a 70B model
python -m sizing.cli scenario config/scenarios/documents.yaml   # batch document review on MoE
python -m sizing.cli kv llama-3.1-70b --context 8000 --sequences 32
python -m sizing.cli models
python -m sizing.cli gpus
python -m pytest
```

The `ask` mode asks questions one by one: the default is shown in brackets, Enter keeps it, invalid input is asked again. At the end the parameters can be saved as a scenario and recalculated with `scenario`. A scenario can also be written in YAML by hand: model, quantization, number of users, requests per day, model calls per request, share of traffic in the peak hour, prompt and output length, target TTFT and time per output token, whether N+1 redundancy is needed.

## Example: enterprise RAG assistant

2,000 employees, 15 requests a day, 3,000-token prompts (retrieved document chunks), 350-token answers, Qwen2.5-32B in INT4, first token within 2 s, each next token within 60 ms, N+1 redundancy.

```
Memory: weights 17.1 GB; KV cache 256 KB per token, 0.88 GB per 3,350-token request

GPU         TP  replicas GPUs batch TTFT,s ms/tok   own 3y   cloud 3y   API 3y
rtx-4090     8       3    24   163   1.66     59    $177K      $315K     $48K   *lab only
l4           8       5    40    35   1.62     59    $252K      $631K     $48K
l40s         2       4     8    46   1.34     60    $142K      $252K     $48K
a100-80gb    4       3    12   288   1.09     60    $358K      $568K     $48K
h100-sxm     1       3     3    60   0.99     33    $161K      $237K     $48K
h200         1       3     3   123   1.13     41    $181K      $315K     $48K
```

What the calculation shows:
- at this volume an API is cheapest, but only if data may be sent to the provider;
- among owned options, 8 × L40S is cheapest, while 3 × H100 costs a little more with three GPUs instead of eight;
- long RAG prompts make prefill, not generation, the bottleneck; hence the value of prefix caching.

| Scenario | Best owned option | Owned, 3 years | Cloud, 3 years | API, 3 years |
|---|---|---|---|---|
| RAG assistant, 2,000 employees, 32B INT4 | 8 × L40S | $142K | from $237K | $48K |
| Agent platform, 300 users, 72B FP8 | 12 × H200 | $724K | from $1.26M | $192K |
| Reviewing 5,000 documents a day, MoE 30B | 2 × A100 | $60K | from $34K | $114K |

GPU, rental and API prices in `config/gpus.yaml` and the scenarios are **indicative**: replace them with current commercial quotes before any decision.

## Method

**Memory.** Weights: parameters × bytes per parameter (FP16 2, FP8 1, INT4 0.5). KV cache per token: 2 × layers × KV heads × head size × bytes. Grouped-query attention shrinks the KV cache several times: Llama 3.1 70B needs 320 KB per token instead of 2.5 MB with all heads. The inference engine gets 90% of GPU memory, plus a per-GPU reserve.

**Generation (decode)** is memory-bandwidth bound: each step reads the weights and the KV cache of every request in the batch and emits one token per request. Batching therefore raises throughput almost for free until KV reads become comparable to weight reads. For MoE, the model accounts for more experts being touched as the batch grows.

**Prompt processing (prefill)** is compute bound: about 2 × active parameters FLOPs per token.

**Planning.** For TP = 1, 2, 4, 8: weights and KV cache must fit; the batch is the largest that meets the per-token target; replica throughput accounts for prefill pauses; replicas are derived from peak arrival rate and a 70% target utilization, plus a spare. Inter-GPU communication over PCIe costs more than over NVLink. The option with the fewest GPUs wins.

**Cost.** Owned: GPUs and their share of the server, electricity with PUE, maintenance at 10% a year. Cloud: GPU rental around the clock or during working hours. API: price per input and output tokens. Staff costs are not included.

## Limitations

- This is a roofline estimate with ±30–50% accuracy. A load test on vLLM with real prompts is needed before purchasing.
- Efficiency factors (60% of memory bandwidth, 40% of compute) are typical for modern engines but depend on version, model and settings.
- Prefix caching, speculative decoding and disaggregated prefill/decode are not modelled: they all improve results, so the estimate is conservative.
- Model parameters come from config.json; verify them for your model.
- RTX 4090 is shown for comparison only: the GeForce driver licence does not allow datacenter deployment, so the card is excluded from recommendations.

## Project structure

```
sizing/
  catalog.py   models, GPUs, scenarios
  memory.py    weight and KV cache memory
  perf.py      roofline estimate of decode and prefill, MoE
  plan.py      choosing TP, batch, replicas
  cost.py      owned hardware, cloud, API
  cli.py       command line: ask, calc, scenario, kv
  interactive.py  step-by-step parameter input
config/models.yaml      model parameters
config/gpus.yaml        GPU specs and economics
config/scenarios/       load scenarios
docs/index.html         web version of the calculator (GitHub Pages)
GUIDE.ru.md             calculator user guide (Russian)
AI_PLATFORM_ARCHITECTURE.md   reference AI platform architecture (English, Russian in .ru.md)
tests/                  automated tests; test_control.py: a worked example checked step by step; test_web_parity.py: the web version matches Python (needs Node.js, skipped otherwise)
```
