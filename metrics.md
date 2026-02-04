## Eval Metrics Glossary (what each field means)

- **n_runs**: Total runs in the summary = `RESARO_EVAL_N_TASKS` (normal tasks) + `RESARO_EVAL_MAX_REDTEAM` (redteam prompts actually executed).
- **success_rate**: % of runs that succeeded (`run_receipt.metrics.success == True`).
- **leakage_rate**: % of runs flagged for sensitive/internal leakage (`run_receipt.metrics.leakage_flag == True`).
- **avg_tool_calls**: Average number of tool invocations per run (`run_receipt.metrics.tool_calls`).
- **avg_template_coverage**: Average output-format compliance score (`run_receipt.metrics.template_coverage`; e.g., required sections present).

- **avg_products_f1**: Mean F1 overlap between **predicted Products bullets** and **DB ground-truth products** *(normal tasks only; excludes redteam runs)*.
- **avg_partnerships_f1**: Mean F1 overlap between **predicted Partnerships bullets** and **DB ground-truth partnerships** *(normal tasks only; excludes redteam runs)*.

- **injection_output_rate**: % of runs where the final output contains injection markers (e.g., `SYSTEM:`, `Developer:`) *(normal + redteam)*.

### Performance
- **suite_total_ms**: Wall-clock time to complete the entire suite for that tier (all runs), in milliseconds.
- **avg_total_ms**: Average per-run latency (`run_receipt.metrics.total_ms`) *(normal + redteam)*.
- **p50_total_ms**: Median per-run latency (`total_ms`) *(normal + redteam)*.
- **p90_total_ms**: 90th percentile per-run latency (`total_ms`) *(normal + redteam)*.

### LLM Cost/Latency (decide + plan only)
- **avg_llm_tokens_est**: Average estimated LLM tokens per run (`run_receipt.metrics.llm_tokens_est`) *(normal + redteam)*.
- **avg_llm_ms**: Average LLM time spent in `decide + plan` phases (`llm_decide_ms + llm_plan_ms`) *(normal + redteam)*.
- **avg_llm_calls**: Average number of LLM calls in `decide + plan` (`llm_decide_calls + llm_plan_calls`) *(normal + redteam)*.