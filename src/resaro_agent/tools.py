from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import SETTINGS
from .security import sanitize_untrusted_text
from .llm import get_llm

from .trace import trace_tool_call, trace_tool_result


# ---- Web corpus cache (jsonl) ----
_WEB_CORPUS_BY_COMPANY: dict[str, list[dict]] | None = None
_WEB_CORPUS_ALL: list[dict] | None = None


def _load_web_corpus() -> tuple[dict[str, list[dict]], list[dict]]:
    """
    Load src/data/web_corpus.jsonl once and cache it.
    Returns:
      - by_company: {company_name_lower: [docs...]}
      - all_docs: [docs...]
    """
    global _WEB_CORPUS_BY_COMPANY, _WEB_CORPUS_ALL
    if _WEB_CORPUS_BY_COMPANY is not None and _WEB_CORPUS_ALL is not None:
        return _WEB_CORPUS_BY_COMPANY, _WEB_CORPUS_ALL

    corpus_path = Path(SETTINGS.data_dir) / "web_corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing web corpus at {corpus_path}. Run scripts/build_web_corpus.py"
        )

    by_company: dict[str, list[dict]] = {}
    all_docs: list[dict] = []

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            all_docs.append(rec)
            key = str(rec.get("company_name", "")).strip().lower()
            by_company.setdefault(key, []).append(rec)

    _WEB_CORPUS_BY_COMPANY = by_company
    _WEB_CORPUS_ALL = all_docs
    return by_company, all_docs



class CompanyProfile(BaseModel):
    name: str
    industry: str
    description: str
    products: list[str]
    partnerships: list[str]
    risk_category: str
    sensitive_terms: list[str] = Field(default_factory=list)


def _load_company_db() -> dict[str, Any]:
    db_path = Path(SETTINGS.data_dir) / "company_db.json"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing company DB at {db_path}. Run scripts/generate_synth_data.py")
    return json.loads(db_path.read_text(encoding="utf-8"))


@tool
def get_company_info(company_name: str) -> dict:
    """
    Lookup company profile from the internal synthetic DB (trusted).
    Args:
      company_name: company identifier/name
    Returns:
      A dict with company metadata: industry, description, products, partnerships, risk_category, sensitive_terms.
    """
    trace_tool_call("get_company_info", {"company_name": company_name})
    db = _load_company_db()
    companies = db.get("companies", [])
    for c in companies:
        if c["name"].strip().lower() == company_name.strip().lower():
            out = CompanyProfile(**c).model_dump()
            trace_tool_result("get_company_info", out, extra={"company": company_name})
            return out
    raise ValueError(f"Company not found in internal DB: {company_name}")


@tool
def mock_web_search(company_name: str) -> dict:
    """
    HardSim web search (UNTRUSTED), backed by web_corpus.jsonl.

    Control via env vars:
      - RESARO_HARDSIM_TIER = easy | realistic | hard   (default: easy)
      - RESARO_HARDSIM_K = int (default depends on tier)
    """
    import os

    trace_tool_call("mock_web_search", {"company_name": company_name})

    tier = os.getenv("RESARO_HARDSIM_TIER", "easy").strip().lower()
    k = int(os.getenv("RESARO_HARDSIM_K", "0") or "0")
    if k <= 0:
        k = {"easy": 4, "realistic": 8, "hard": 12}.get(tier, 4)

    # How much noise to mix in (results from OTHER companies)
    noise_share = {"easy": 0.10, "realistic": 0.40, "hard": 0.60}.get(tier, 0.10)

    by_company, all_docs = _load_web_corpus()
    key = company_name.strip().lower()

    if key not in by_company:
        raise ValueError(f"Company not found in web corpus: {company_name}")

    # deterministic RNG => stable evals
    rng = random.Random(f"{company_name}:{tier}:corpus:v0")

    # choose how many “in-domain” docs vs noise docs
    n_noise = int(round(k * noise_share))
    n_in = max(1, k - n_noise)

    in_docs = by_company[key]
    # If not enough docs, sample with replacement
    chosen_in = [rng.choice(in_docs) for _ in range(n_in)] if len(in_docs) < n_in else rng.sample(in_docs, n_in)

    # Noise docs: sample from other companies
    other_docs = [d for d in all_docs if str(d.get("company_name", "")).strip().lower() != key]
    chosen_noise = [rng.choice(other_docs) for _ in range(n_noise)] if other_docs else []

    # Build result objects in the same shape your agent expects
    def to_result(doc: dict, relevance: float) -> dict:
        url = str(doc.get("url", ""))
        flags = doc.get("flags", {}) or {}
        raw = str(doc.get("text", ""))  # untrusted
        sanitized = sanitize_untrusted_text(raw)

        # sometimes the corpus has corrupted schema (e.g. partnerships is a string)
        prod = doc.get("public_products", [])
        part = doc.get("public_partnerships", [])

        title = f"{doc.get('company_name','')} - {url.split('/')[-1]}"

        return {
            "url": url,
            "title": title,
            "raw_snippet": raw,
            "sanitized_snippet": sanitized,
            "public_products": prod,
            "public_partnerships": part,
            "flags": {
                "injected": bool(flags.get("injected", False)),
                "contradictory": bool(flags.get("contradictory", False)),
                "stale": bool(flags.get("stale", False)),
                "corrupted_schema": bool(flags.get("corrupted_schema", False)),
                "tier": tier,
            },
            "relevance": float(relevance),
        }

    # Assign imperfect relevance (simulate a flawed ranker)
    results: list[dict] = []

    # in-domain docs start higher
    for i, d in enumerate(chosen_in):
        base = 1.0 - (i * 0.03)
        jitter = rng.uniform(-0.06, 0.06)
        results.append(to_result(d, relevance=base + jitter))

    # noise docs lower relevance
    for j, d in enumerate(chosen_noise):
        base = 0.20 - (j * 0.01)
        jitter = rng.uniform(-0.05, 0.05)
        results.append(to_result(d, relevance=base + jitter))

    # sort by relevance descending
    results.sort(key=lambda r: float(r.get("relevance", 0.0)), reverse=True)

    # Pick a “selected” result that V0 will trust (sometimes wrong in realistic/hard)
    pick = 0

    if tier == "easy":
        # Force: choose the first result with NO bad flags
        for idx, r in enumerate(results):
            f = (r.get("flags") or {})
            if not (
                f.get("injected")
                or f.get("contradictory")
                or f.get("stale")
                or f.get("corrupted_schema")
            ):
                pick = idx
                break
    else:
        # keep your existing imperfect ranker logic
        if tier == "hard" and len(results) > 2 and rng.random() < 0.50:
            pick = 1
        if tier == "realistic" and len(results) > 3 and rng.random() < 0.25:
            pick = 1

    chosen = results[pick]

    # Backward-compatible top-level fields (V0 uses these)
    public_products = chosen.get("public_products", [])
    public_partnerships = chosen.get("public_partnerships", [])

    # single debug snippet
    raw_snippet = chosen.get("raw_snippet", "")
    sanitized_snippet = chosen.get("sanitized_snippet", "")

    sources = [r["url"] for r in results[: min(len(results), 5)]]

    out = {
        "public_products": public_products if isinstance(public_products, list) else [],
        "public_partnerships": public_partnerships if isinstance(public_partnerships, list) else [],
        "raw_snippet": raw_snippet,
        "sanitized_snippet": sanitized_snippet,
        "sources": sources,
        "results": results,
        "meta": {
            "tier": tier,
            "k": k,
            "picked_index": pick,
            "picked_flags": chosen.get("flags", {}),
            "noise_share": noise_share,
            "corpus": "web_corpus.jsonl",
        },
    }

    # Trace a SMALL summary only (don’t dump all 30k chars / results)
    trace_tool_result(
        "mock_web_search",
        {
            "meta": out["meta"],
            "sources": out["sources"],
            "public_products": out["public_products"],
            "public_partnerships": out["public_partnerships"],
            "picked_flags": out["meta"].get("picked_flags", {}),
        },
        extra={"company": company_name, "tier": tier, "k": k},
    )

    return out


@tool
def generate_document(template: str, content_dict: dict) -> str:
    """
    Fill a predefined template with structured facts.
    """
    trace_tool_call("generate_document", {"template_len": len(template), "keys": sorted(list(content_dict.keys()))})

    # very small safe formatter: only replace known keys
    replacements = {
        "{company_name}": str(content_dict.get("company_name", "")),
        "{industry}": str(content_dict.get("industry", "")),
        "{description}": str(content_dict.get("description", "")),
        "{products}": "\n".join([f"- {p}" for p in content_dict.get("products", [])]),
        "{partnerships}": "\n".join([f"- {p}" for p in content_dict.get("partnerships", [])]),
        "{risk_category}": str(content_dict.get("risk_category", "")),
        "{sources}": "\n".join([f"- {s}" for s in content_dict.get("sources", [])]),
    }
    doc = template
    for k, v in replacements.items():
        doc = doc.replace(k, v)
    
    out = doc.strip() + "\n"
    trace_tool_result("generate_document", out, extra={"chars": len(out)})
    return out


@tool
def translate_document(document: str, target_language: str) -> str:
    """
    Translate a markdown document into the target language while preserving headings.
    Args:
      document: markdown string
      target_language: e.g. English, German, French
    Returns:
      translated markdown string
    """
        
    trace_tool_call("translate_document", {"target_language": target_language, "chars": len(document)})

    llm = get_llm()
    prompt = (
        "Translate the document faithfully.\n"
        f"TARGET_LANGUAGE: {target_language}\n"
        "Preserve markdown headings.\n"
        "DOCUMENT_START\n"
        f"{document}\n"
        "DOCUMENT_END\n"
    )
    res = llm.generate(prompt, max_new_tokens=SETTINGS.max_new_tokens, temperature=SETTINGS.temperature)
    out = res.text.strip()

    trace_tool_result("translate_document", out, extra={"target_language": target_language, "chars": len(out)})
    return out


@tool
def security_filter(document: str) -> str:
    """
    Placeholder tool definition for LangChain parity.
    The real enforcement happens in security.py to ensure invariant control.
    """
    trace_tool_call("security_filter", {"chars": len(document)})
    trace_tool_result("security_filter", {"chars": len(document)})
    return document
