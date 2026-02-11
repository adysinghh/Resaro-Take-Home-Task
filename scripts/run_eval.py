"""
scripts.run_eval
Otchestrator/ wrapper for eval_v0.py
Sets tier env, prints save combined summary and also specifies the number of Redteam attacks to be done 'max_redteam'
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.resaro_agent.eval_v0 import run_suite


def main():
    Path("reports").mkdir(exist_ok=True)

    tiers = ["easy", "realistic", "hard"]
    summaries = {}

    max_redteam = int(os.getenv("RESARO_EVAL_MAX_REDTEAM", "0"))  # <-- default 0

    for tier in tiers:
        os.environ["RESARO_HARDSIM_TIER"] = tier
        out_dir = f"reports/run_logs_{tier}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        result = run_suite(out_dir=out_dir, max_redteam=max_redteam)  # <-- uses env
        summaries[tier] = result["summary"]

        print(f"\n=== V0 EVAL SUMMARY ({tier.upper()}) ===")
        print(json.dumps(result["summary"], indent=2))

    Path("reports", "summary_all_tiers.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print("\nWrote combined summary to reports/summary_all_tiers.json")


if __name__ == "__main__":
    main()