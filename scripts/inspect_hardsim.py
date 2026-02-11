"""
scripts.inspect_hardsim

Runs company queries across easy/realistic/hard HardSim tiers and logs retrieval metadata
so we can validate whether aggregation ever falls back to the picked (possibly noisy) doc.

Default: scans first 1000 companies (indices 0..999) and saves results to a .txt file
for easy cmd+f (search for "fallback_to_chosen").

Run - 'export RESARO_AGG_TOP_N=1' for changeing the top_n to 1 for no aggregation and choose more infected docs
"""

import os
import json
from pathlib import Path
from datetime import datetime

from src.resaro_agent.config import SETTINGS
from src.resaro_agent.tools import mock_web_search


TIERS = ["easy", "realistic", "hard"]


def load_companies() -> list[dict]:
    db_path = Path(SETTINGS.data_dir) / "company_db.json"
    db = json.loads(db_path.read_text(encoding="utf-8"))
    return db.get("companies", [])


def run_one(company_name: str, tier: str) -> dict:
    os.environ["RESARO_HARDSIM_TIER"] = tier
    out = mock_web_search.invoke({"company_name": company_name})
    meta = out.get("meta", {}) or {}
    agg = meta.get("aggregation", {}) or {}

    # Minimal structured summary (good for scanning + grep/cmd+f)
    return {
        "tier": tier,
        "company": company_name,
        "meta": meta,
        "aggregation": agg,
        "fallback_to_chosen": bool(agg.get("fallback_to_chosen", False)),
        "picked_flags": meta.get("picked_flags", {}),
        "picked_index": meta.get("picked_index", None),
        "k": meta.get("k", None),
        "noise_share": meta.get("noise_share", None),
        "used_indexes": agg.get("used_indexes", []),
        # keep a tiny preview for context (optional)
        "raw_snippet_preview": (out.get("raw_snippet", "") or "")[:140].replace("\n", "\\n"),
        # top 3 flags snapshot
        "top3": [
            {"url": r.get("url"), "flags": r.get("flags")}
            for r in (out.get("results", []) or [])[:3]
        ],
    }


def main(
    start_idx: int = 0,
    end_idx_exclusive: int = 1000,
    out_file: str | None = None,
):
    companies = load_companies()
    if not companies:
        raise RuntimeError("No companies found in company_db.json")

    # clamp range safely
    start_idx = max(0, start_idx)
    end_idx_exclusive = min(len(companies), max(start_idx + 1, end_idx_exclusive))

    if out_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"artifacts/hardsim_scan_{start_idx}_{end_idx_exclusive-1}_{ts}.txt"

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # counters for quick summary
    total_runs = 0
    fallback_true = 0
    fallback_true_by_tier = {t: 0 for t in TIERS}

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"HardSim scan log\n")
        f.write(f"DB: {Path(SETTINGS.data_dir) / 'company_db.json'}\n")
        f.write(f"Range: [{start_idx}, {end_idx_exclusive})  (count={end_idx_exclusive-start_idx})\n")
        f.write(f"Tiers: {TIERS}\n")
        f.write("=" * 90 + "\n\n")

        for idx in range(start_idx, end_idx_exclusive):
            company_name = companies[idx].get("name", f"<missing-name-{idx}>")

            for tier in TIERS:
                total_runs += 1
                summary = run_one(company_name, tier)

                # detect fallback
                if summary["fallback_to_chosen"]:
                    fallback_true += 1
                    fallback_true_by_tier[tier] += 1

                # write a human-readable block (best for cmd+f)
                f.write("-" * 90 + "\n")
                f.write(f"IDX={idx}  COMPANY={company_name}  TIER={tier.upper()}\n")
                f.write(f"fallback_to_chosen: {summary['fallback_to_chosen']}\n")
                f.write(f"k={summary['k']}  noise_share={summary['noise_share']}  picked_index={summary['picked_index']}\n")
                f.write(f"picked_flags={summary['picked_flags']}\n")
                f.write(f"used_indexes={summary['used_indexes']}\n")
                f.write(f"aggregation={summary['aggregation']}\n")
                f.write(f"raw_snippet_preview={summary['raw_snippet_preview']} …\n")
                f.write(f"top3={summary['top3']}\n\n")

        # footer summary
        f.write("=" * 90 + "\n")
        f.write("SUMMARY\n")
        f.write(f"total_runs={total_runs}\n")
        f.write(f"fallback_true_total={fallback_true}\n")
        f.write(f"fallback_true_by_tier={fallback_true_by_tier}\n")

    print(f"Wrote scan log -> {out_path}")
    print(f"total_runs={total_runs} fallback_true_total={fallback_true} fallback_true_by_tier={fallback_true_by_tier}")
    print('Tip: cmd+f in the txt for "fallback_to_chosen: True"')


if __name__ == "__main__":
    # default: scan first 1000 companies (0..999)
    main(start_idx=0, end_idx_exclusive=1000)
