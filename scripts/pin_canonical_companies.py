from __future__ import annotations
import json
from pathlib import Path

PINNED = [
    "Asteron", "Novacrest", "HelioForge", "Quartzline", "Boreal Dynamics",
    "MiraWorks", "Syntera Labs", "CobaltWave", "Pinebridge Systems", "OrchidShield",
]

def rewrite_prefix(text: str, old: str, new: str) -> str:
    if not text:
        return text
    return text.replace(old, new, 1)

def main():
    db_path = Path("src/data/company_db.json")
    db = json.loads(db_path.read_text(encoding="utf-8"))
    companies = db.get("companies", [])

    if len(companies) < len(PINNED):
        raise RuntimeError(f"DB has only {len(companies)} companies; need at least {len(PINNED)}")

    for i, new_name in enumerate(PINNED):
        c = companies[i]
        old_name = c["name"]

        # rename company
        c["name"] = new_name

        # rewrite description prefix (if it contains the old name)
        c["description"] = rewrite_prefix(c.get("description", ""), old_name, new_name)

        # rewrite product names that are prefixed with old company name
        prods = c.get("products", []) or []
        c["products"] = [rewrite_prefix(p, old_name, new_name) for p in prods]

        # partnerships usually don’t include company name, leave as-is

    db_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(f"Pinned canonical 10 names into {db_path}")

if __name__ == "__main__":
    main()
