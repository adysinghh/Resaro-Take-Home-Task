"""
Docstring for scripts.pin_canonical_companies

post-processing fixer for the company_db.json
script is effectively a safety/idempotency step
Updated - 0-9 companies
    i. descp. prefix
   ii. product name prefix
  iii. leaves the partnership

"""

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

    # Load the json
    db_path = Path("src/data/company_db.json")
    db = json.loads(db_path.read_text(encoding="utf-8"))
    companies = db.get("companies", [])

    # checks if 'company_db' has at least 10 companies
    if len(companies) < len(PINNED):
        raise RuntimeError(f"DB has only {len(companies)} companies; need at least {len(PINNED)}")

    # for the first 10 companies, rename them; replaces 'companies[i]["name"]' with 'PINNED[i]'
    for i, new_name in enumerate(PINNED):
        c = companies[i]
        old_name = c["name"]

        # rename company
        c["name"] = new_name

        # updates the first occurrence of old name in description.
        c["description"] = rewrite_prefix(c.get("description", ""), old_name, new_name)

        # updates first occurrence of old name in each product string
        prods = c.get("products", []) or []
        c["products"] = [rewrite_prefix(p, old_name, new_name) for p in prods]

        # partnerships usually don’t include company name, leave as-is

    # writes the file back.
    db_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(f"Pinned canonical 10 names into {db_path}")

if __name__ == "__main__":
    main()
