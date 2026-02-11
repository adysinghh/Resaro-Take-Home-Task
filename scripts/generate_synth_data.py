"""
scripts.generate_synth_data
Local simulated company dataset
OUTPUT: company_json.db
{
  "companies": [
    {
      "name": "...",
      "industry": "...",
      "description": "...",
      "products": ["...", "...", "..."],
      "partnerships": ["..."],
      "risk_category": "...",
      "sensitive_terms": ["..."]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Hardcoded INDUSTRIES list
INDUSTRIES = [
    "FinTech", "HealthTech", "Energy", "Logistics", "Cybersecurity", "Retail AI",
    "DevTools", "InsurTech", "ClimateTech", "EdTech", "LegalTech", "BioTech",
]

# Hardcoded RISK list
RISK = ["low", "medium", "high"]


# Hardcoded Sensitives terms
SENSITIVE_POOL = [
    "Project Nightfall", "Apollo-X", "Internal Codename Kappa", "Operation Glassbox",
    "Confidential Initiative Orion", "Project Red Maple", "Project IonWisp",
    "Delta-Signal", "Orchid Protocol", "Sable Initiative", "Kestrel Vault",
]

# Hardcoded Partners
PARTNERS_POOL = [
    "Globex", "Initech", "Umbrella", "Stark Industries", "Wayne Enterprises",
    "Soylent", "Tyrell", "Wonka Industries", "Hooli", "Vehement Capital",
]

# Hardcoded Product Suffixes
PRODUCT_SUFFIXES = ["Cloud", "Analytics", "Assist", "Shield", "Edge", "Flow", "Vault", "AI", "Ops", "Secure"]

SYLLABLES_A = ["As", "No", "He", "Qua", "Bo", "Mi", "Syn", "Co", "Pine", "Or", "Ze", "Lu", "Va", "Tri", "Mar", "Sol"]
SYLLABLES_B = ["ter", "va", "lio", "artz", "re", "ra", "te", "balt", "bridge", "chid", "phyr", "men", "dor", "gen", "nix", "vex"]
SYLLABLES_C = ["on", "crest", "forge", "line", "dynamics", "works", "labs", "wave", "systems", "shield", "logic", "stack", "nova", "point", "gate", "core"]

# 10 Pinned company
PINNED = [
    "Asteron", "Novacrest", "HelioForge", "Quartzline", "Boreal Dynamics",
    "MiraWorks", "Syntera Labs", "CobaltWave", "Pinebridge Systems", "OrchidShield",
]


def gen_company(name: str, rng: random.Random) -> dict:
    """
    rng: random choice function from the preceeding list
    """

    industry = rng.choice(INDUSTRIES)

    suffixes = rng.sample(PRODUCT_SUFFIXES, 3)
    # for example - name = "Asteron" * suffixes = ["Cloud", "Analytics", "Shield"] -> products = ["Asteron Cloud", "Asteron Analytics", "Asteron Shield"]
    products = [f"{name} {s}" for s in suffixes] # s -> PRODUCT_SUFFIXES

    # partnerships is just a list of partner company names that the generated company “works with”
    # A company can have maximum 3 partners
    partners = [rng.choice(PARTNERS_POOL) for _ in range(rng.randint(1, 3))] # random choosing 1,2 or 3
    partnerships = list(dict.fromkeys(partners)) # removes duplicates so the final list is unique

    
    risk = rng.choices(RISK, weights=[0.45, 0.35, 0.20], k=1)[0] # weights = [low, medium, high], In many real systems, low-risk entities are more common than high-risk ones
    sensitive_terms = rng.sample(SENSITIVE_POOL, k=2 if risk != "low" else 1) # picks k unique items from the list, if risk is not low then pick 2 items from the list

    return {
        "name": name,
        "industry": industry,
        "description": f"{name} is a {industry} company focused on delivering practical products for enterprise clients.",
        "products": products,
        "partnerships": partnerships,
        "risk_category": risk, # [low, med, high]
        "sensitive_terms": sensitive_terms, # [dependent on the 'risk_category']
    }

# Simple func. which generates rest 990 companies; AcmeSynth + (0-989), eg: AcmeSynth0000 to AcmeSynth989
def _gen_extra_name(i: int) -> str:
    # simple stable extra name generator (readable + unique)
    return f"AcmeSynth{i:04d}"


def main():

    out_dir = Path("src/data")
    out_dir.mkdir(parents=True, exist_ok=True) # creates missing folder, if needed

    rng = random.Random(42)  # seed for reproducibility; remove seed if you want new every run

    TARGET_N = 1000  # for number of companies (10 + 990 = TARGET_N)

    names = PINNED[:]  # keep canonical ones

    # for generaing unique names
    i = 0
    seen = {n.strip().lower() for n in names}

    while len(names) < TARGET_N:
        cand = _gen_extra_name(i)
        i += 1
        if cand.strip().lower() in seen:
            continue
        names.append(cand)
        seen.add(cand.strip().lower())

    companies = [gen_company(n, rng) for n in names]
    payload = {"companies": companies} # {"companies": []}

    (out_dir / "company_db.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote {len(companies)} synthetic companies to src/data/company_db.json")

if __name__ == "__main__":
    main()
