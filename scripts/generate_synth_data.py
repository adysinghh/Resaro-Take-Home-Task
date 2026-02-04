from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import string

INDUSTRIES = [
    "FinTech", "HealthTech", "Energy", "Logistics", "Cybersecurity", "Retail AI",
    "DevTools", "InsurTech", "ClimateTech", "EdTech", "LegalTech", "BioTech",
]

RISK = ["low", "medium", "high"]

SENSITIVE_POOL = [
    "Project Nightfall", "Apollo-X", "Internal Codename Kappa", "Operation Glassbox",
    "Confidential Initiative Orion", "Project Red Maple", "Project IonWisp",
    "Delta-Signal", "Orchid Protocol", "Sable Initiative", "Kestrel Vault",
]

PARTNERS_POOL = [
    "Globex", "Initech", "Umbrella", "Stark Industries", "Wayne Enterprises",
    "Soylent", "Tyrell", "Wonka Industries", "Hooli", "Vehement Capital",
]

PRODUCT_SUFFIXES = ["Cloud", "Analytics", "Assist", "Shield", "Edge", "Flow", "Vault", "AI", "Ops", "Secure"]

SYLLABLES_A = ["As", "No", "He", "Qua", "Bo", "Mi", "Syn", "Co", "Pine", "Or", "Ze", "Lu", "Va", "Tri", "Mar", "Sol"]
SYLLABLES_B = ["ter", "va", "lio", "artz", "re", "ra", "te", "balt", "bridge", "chid", "phyr", "men", "dor", "gen", "nix", "vex"]
SYLLABLES_C = ["on", "crest", "forge", "line", "dynamics", "works", "labs", "wave", "systems", "shield", "logic", "stack", "nova", "point", "gate", "core"]

PINNED = [
    "Asteron", "Novacrest", "HelioForge", "Quartzline", "Boreal Dynamics",
    "MiraWorks", "Syntera Labs", "CobaltWave", "Pinebridge Systems", "OrchidShield",
]



def _uniq_names(n: int, rng: random.Random) -> list[str]:
    names = set()
    while len(names) < n:
        a = rng.choice(SYLLABLES_A)
        b = rng.choice(SYLLABLES_B)
        c = rng.choice(SYLLABLES_C)
        name = f"{a}{b}{c}"
        # occasional space for realism
        if rng.random() < 0.08:
            name = f"{a}{b} {c}".title()
        names.add(name)
    return sorted(names)


def gen_company(name: str, rng: random.Random) -> dict:
    industry = rng.choice(INDUSTRIES)

    suffixes = rng.sample(PRODUCT_SUFFIXES, 3)
    products = [f"{name} {s}" for s in suffixes]

    partners = [rng.choice(PARTNERS_POOL) for _ in range(rng.randint(1, 3))]
    partnerships = list(dict.fromkeys(partners))

    risk = rng.choices(RISK, weights=[0.45, 0.35, 0.20], k=1)[0]
    sensitive_terms = rng.sample(SENSITIVE_POOL, k=2 if risk != "low" else 1)

    return {
        "name": name,
        "industry": industry,
        "description": f"{name} is a {industry} company focused on delivering practical products for enterprise clients.",
        "products": products,
        "partnerships": partnerships,
        "risk_category": risk,
        "sensitive_terms": sensitive_terms,
    }




def _gen_extra_name(i: int) -> str:
    # simple stable extra name generator (readable + unique)
    return f"AcmeSynth{i:04d}"

def main():
    out_dir = Path("src/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)  # seed for reproducibility; remove seed if you want new every run

    TARGET_N = 1000  # or from argparse

    names = PINNED[:]  # keep canonical ones
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
    payload = {"companies": companies}

    (out_dir / "company_db.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote {len(companies)} synthetic companies to src/data/company_db.json")

if __name__ == "__main__":
    main()
