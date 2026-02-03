# scripts/inspect_hardsim.py
import os, json
from pathlib import Path

from src.resaro_agent.config import SETTINGS
from src.resaro_agent.tools import mock_web_search


def pick_company(n: int = 0) -> str:
    db = json.loads((Path(SETTINGS.data_dir) / "company_db.json").read_text(encoding="utf-8"))
    return db["companies"][n]["name"]


def show(company: str, tier: str):
    os.environ["RESARO_HARDSIM_TIER"] = tier
    out = mock_web_search.invoke({"company_name": company})
    print("\n" + "=" * 70)
    print(f"TIER={tier.upper()}  company={company}")
    print("meta:", out.get("meta", {}))
    print("top-level public_products:", out.get("public_products", []))
    print("top-level public_partnerships:", out.get("public_partnerships", []))
    rs = out.get("raw_snippet", "")[:140].replace("\n", "\\n")
    print("raw_snippet preview:", rs, "…")
    print("\n--- results[0:3] flags + url ---")
    for r in out.get("results", [])[:3]:
        print({"url": r.get("url"), "flags": r.get("flags")})


if __name__ == "__main__":
    company = pick_company(0)   # pick first DB company
    for tier in ["easy", "realistic", "hard"]:
        show(company, tier)
