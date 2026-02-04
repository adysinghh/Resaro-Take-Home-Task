### `README.md`

````md
# Resaro Take-home: Agentic LLM + Testing

This repo contains:
- `writeup.md` (build journey + design)
- `metrics.md` (eval metrics and what they mean)
- `test_plan.md` (test plan and checks)
- `Research_Referenced/` (papers and notes used while designing the system)
````

## `Quickstart`

### Option A: Run UI (recommended)
Easy setup. UI includes buttons for bootstrap and validation.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r req.txt
cp .env.example .env

export RESARO_LLM_BACKEND=local
streamlit run ui/app.py
````

### Option B: Run from terminal

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r req.txt
cp .env.example .env

export RESARO_LLM_BACKEND=local
python scripts/generate_synth_data.py
python scripts/run_eval.py
```

## `Data Setup (HardSim)`

I used fully synthetic data to evaluate the agent in controlled HardSim conditions:

* `src/data/company_db.json` - synthetic internal company database (ground truth)
* `src/data/web_corpus.jsonl` - synthetic “web pages” corpus used by `mock_web_search` (untrusted)

```bash
# ensure local imports work
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 1) Generate synthetic company DB (ground truth)
# Creates src/data/company_db.json with 1000 companies (first 10 are canonical demo names)
python scripts/generate_synth_data.py
python scripts/pin_canonical_companies.py

# 2) Generate synthetic web corpus (mock web)
# Creates src/data/web_corpus.jsonl (30,000 docs by default)
python scripts/build_web_corpus.py \
  --db src/data/company_db.json \
  --out src/data/web_corpus.jsonl \
  --pages_per_company 30 \
  --seed 7

# Note: pages_per_company=30 × 1000 companies = 30,000 synthetic web pages
```

## `Data Verification`

```bash
# Inspect HardSim tiers (easy / realistic / hard)
# Shows how the same company search differs across tiers (k, noise share, injected/contradictory/stale docs)
python scripts/inspect_hardsim.py
```

Corpus flag distribution:

```bash
python - <<'PY'
import json, collections
from pathlib import Path

p = Path("src/data/web_corpus.jsonl")
n = 0
flags = collections.Counter()

for line in p.open(encoding="utf-8"):
    n += 1
    rec = json.loads(line)
    fl = rec.get("flags", {}) or {}
    for k, v in fl.items():
        if v:
            flags[k] += 1

print("docs:", n)
for k, v in flags.most_common():
    print(k, f"{v/n:.3f}")
PY
```

## `Eval results (v2, 50 runs)`

| Tier      | n_runs | success_rate | leakage_rate | avg_tool_calls | avg_template_coverage | avg_products_f1 | avg_partnerships_f1 | injection_output_rate | suite_total_ms | avg_total_ms | p50_total_ms | p90_total_ms | avg_llm_tokens_est | avg_llm_ms | avg_llm_calls |
| --------- | -----: | -----------: | -----------: | -------------: | --------------------: | --------------: | ------------------: | --------------------: | -------------: | -----------: | -----------: | -----------: | -----------------: | ---------: | ------------: |
| EASY      |     50 |         0.98 |          0.0 |          12.20 |                   1.0 |          0.8837 |              0.9049 |                   0.0 |         827858 |     16540.18 |      11307.0 |        21997 |            2448.98 |    7429.28 |          4.38 |
| REALISTIC |     50 |         1.00 |          0.0 |          12.18 |                   1.0 |          0.8240 |              0.8971 |                   0.0 |         792202 |     15830.32 |      11348.5 |        22917 |            2449.54 |    7505.00 |          4.38 |
| HARD      |     50 |         0.98 |          0.0 |          12.24 |                   1.0 |          0.7189 |              0.8126 |                   0.0 |        1570689 |     31398.80 |      11452.0 |        23842 |            2457.92 |    7545.62 |          4.38 |

Notes:

1. Formatting and safety are stable (template coverage 1.0, leakage 0.0 across tiers).
2. Extraction accuracy drops on HARD as expected due to noisy or contradictory evidence, reflected in lower F1.
3. HARD tier latency is higher (avg ~31s), likely due to more fixes or heavier processing under difficult conditions.

## Where to look (artifacts)

* Eval logs: `reports/run_logs/`
* Per-task output markdown: `reports/run_logs/*_final.md`
* Traces (if enabled): `reports/traces/`

## Research referenced

| Bucket                            | Papers Referenced                                                                | Why (1-liner)                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Agents - Basic                    | **AGENT S**; **AGENTSTUDIO**; **Automated Design of Agentic Systems**; **ReAct** | Baseline agent loop + tool-use orchestration patterns; how modern agent systems structure control and evaluation. |
| Memory                            | **Dynamic Cheatsheet**; **SimpleMem**                                            | Practical memory primitives: store what worked, retrieve under constraints, reduce repeated fixes and latency.    |
| Planning & Optimality             | **Plan-and-Act**; **ReflAct**; **Tree of Thoughts**                              | Better control: stage completeness, reflection signals to prevent loops; optional structured search ideas.        |
| Prompt Evolution / Self-improving | **SCOPE**                                                                        | Across-episodes improvements via prompt evolution and guidelines accumulation.                                    |
| Reward Model & RL                 | **Toolcall Reward Model**; **ToolRL**; **ToolRM**                                | Reward shaping for tool-choice efficiency and correctness; future path to reduce redundant tool calls.            |

---
## UI

1. Home View:

![UI home](ui/assets/ui_home.png)

2. Document Preview After Run:

![UI eval](ui/assets/ui_doc.png)