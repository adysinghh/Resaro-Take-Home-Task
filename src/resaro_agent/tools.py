from __future__ import annotations


import re
import os
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

# ---- HardSim defaults ----
_DEFAULT_K_BY_TIER = {"easy": 4, "realistic": 8, "hard": 12}
_NOISE_SHARE_BY_TIER = {"easy": 0.10, "realistic": 0.40, "hard": 0.60}

# Pick logic (probability of trusting 2nd ranked doc)
_PICK_SECOND_PROB = {"hard": 0.50, "realistic": 0.25}

# Aggregation defaults
_DEFAULT_AGG_TOP_N = 5 # change this to 1 for more robust agent test 
_TRACE_SOURCES_N = 5



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


# Build result objects in the same shape your agent expects
def _to_search_result(doc: dict, *, tier: str, relevance: float) -> dict:
    url = str(doc.get("url", ""))
    flags = doc.get("flags", {}) or {}
    raw = str(doc.get("text", ""))  # untrusted
    sanitized = sanitize_untrusted_text(raw)

    # sometimes the corpus has corrupted schema (e.g. products/partnerships is a string)
    raw_prod = doc.get("public_products", [])
    raw_part = doc.get("public_partnerships", [])

    prod = raw_prod if isinstance(raw_prod, list) else []
    part = raw_part if isinstance(raw_part, list) else []

    corrupted_by_type = (not isinstance(raw_prod, list)) or (not isinstance(raw_part, list))


    title = f"{doc.get('company_name','')} - {url.split('/')[-1]}"

    return {
        "url": url,
        "title": title,
        "company_name": str(doc.get("company_name", "")).strip(),
        "raw_snippet": raw,
        "sanitized_snippet": sanitized,
        "public_products": prod,
        "public_partnerships": part,
        "flags": {
            "injected": bool(flags.get("injected", False)),
            "contradictory": bool(flags.get("contradictory", False)),
            "stale": bool(flags.get("stale", False)),
            "corrupted_schema": bool(flags.get("corrupted_schema", False)) or corrupted_by_type,
            "tier": tier,
        },
        "relevance": float(relevance),
    }


def _get_hardsim_params() -> tuple[str, int, float]:
    tier = os.getenv("RESARO_HARDSIM_TIER", "easy").strip().lower()
    k = int(os.getenv("RESARO_HARDSIM_K", "0") or "0")
    if k <= 0:
        k = _DEFAULT_K_BY_TIER.get(tier, 4)
    noise_share = _NOISE_SHARE_BY_TIER.get(tier, 0.10)
    return tier, k, noise_share

def _sample_docs(rng: random.Random, in_docs: list[dict], other_docs: list[dict], *, k: int, noise_share: float) -> tuple[list[dict], list[dict]]:
    n_noise = int(round(k * noise_share))
    n_in = max(1, k - n_noise)

    chosen_in = [rng.choice(in_docs) for _ in range(n_in)] if len(in_docs) < n_in else rng.sample(in_docs, n_in)
    chosen_noise = [rng.choice(other_docs) for _ in range(n_noise)] if other_docs else []
    return chosen_in, chosen_noise

def _aggregate_facts(
    *,
    results: list[dict],
    key: str,
    tier: str,
    top_n: int,
    chosen: dict,
) -> tuple[list[str], list[str], dict]:
    """
    Aggregate public_products/public_partnerships across top-N results, tier-aware.
    Returns:
      - public_products (list[str])
      - public_partnerships (list[str])
      - aggregation_meta (dict)  [same keys as before]
    """
    top_n = max(1, min(top_n, len(results)))
    candidates = results[:top_n]

    def _is_agg_allowed(r: dict) -> bool:
        f = (r.get("flags") or {})

        # Always reject these
        if f.get("injected") or f.get("corrupted_schema"):
            return False

        # Only aggregate in-domain docs (avoid noise-company pollution)
        r_company = str(r.get("company_name", "")).strip().lower()
        if r_company != key:
            return False

        # Tier-dependent strictness
        if tier == "easy":
            if f.get("stale") or f.get("contradictory"):
                return False
        elif tier == "realistic":
            if f.get("stale"):
                return False
        # hard: allow stale/contradictory (but still reject injected/corrupt)

        return True

    agg_products: set[str] = set()
    agg_partners: set[str] = set()
    used_idxs: list[int] = []

    for idx, r in enumerate(candidates):
        if not _is_agg_allowed(r):
            continue

        used_idxs.append(idx)

        prods = r.get("public_products", [])
        parts = r.get("public_partnerships", [])

        if isinstance(prods, list):
            for p in prods:
                if isinstance(p, str) and p.strip():
                    agg_products.add(p.strip())

        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, str) and p.strip():
                    agg_partners.add(p.strip())

    # Fallback to chosen doc if aggregation yields nothing (keeps behavior stable)
    chosen_products = chosen.get("public_products", [])
    chosen_partners = chosen.get("public_partnerships", [])

    public_products = sorted(agg_products) if agg_products else (
        chosen_products if isinstance(chosen_products, list) else []
    )
    public_partnerships = sorted(agg_partners) if agg_partners else (
        chosen_partners if isinstance(chosen_partners, list) else []
    )

    aggregation_meta = {
        "enabled": True,
        "top_n": top_n,
        "used_indexes": used_idxs,
        "used_n": len(used_idxs),
        "agg_products_n": len(public_products),
        "agg_partnerships_n": len(public_partnerships),
        "fallback_to_chosen": (len(used_idxs) == 0),
    }

    return public_products, public_partnerships, aggregation_meta


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
    
    trace_tool_call("mock_web_search", {"company_name": company_name})
    
    # ---- Params & corpus load ----
    # k = number of retrieved docs
    # noise_share = fraction of docs from other companies (noise)
    tier, k, noise_share = _get_hardsim_params()

    by_company, all_docs = _load_web_corpus()
    key = company_name.strip().lower()

    if key not in by_company:
        raise ValueError(f"Company not found in web corpus: {company_name}")

    # deterministic RNG => stable evals
    # ---- Deterministic RNG (stable evals) ----
    rng = random.Random(f"{company_name}:{tier}:corpus:v0")

    # ---- Sample in-domain + noise docs ----
    # choose how many “in-domain” docs vs noise docs
    
    """
        k = total results returned
        noise_share = fraction of results that come from other companies (irrelevant)
        n_in = results from the correct company
        n_noise = irrelevant results from other companies
    """
    n_noise = int(round(k * noise_share))
    n_in = max(1, k - n_noise)

    in_docs = by_company[key]
    # If not enough docs, sample with replacement
    chosen_in = [rng.choice(in_docs) for _ in range(n_in)] if len(in_docs) < n_in else rng.sample(in_docs, n_in)

    # Noise docs: sample from other companies
    other_docs = [d for d in all_docs if str(d.get("company_name", "")).strip().lower() != key]
    chosen_noise = [rng.choice(other_docs) for _ in range(n_noise)] if other_docs else []

    # Assign imperfect relevance (simulate a flawed ranker)
    # ---- Build ranked results (imperfect ranker simulation) ----
    results: list[dict] = []

    # in-domain docs start higher
    for i, d in enumerate(chosen_in):
        base = 1.0 - (i * 0.03)
        jitter = rng.uniform(-0.06, 0.06)
        results.append(_to_search_result(d, tier=tier, relevance=base + jitter))

    # noise docs lower relevance
    for j, d in enumerate(chosen_noise):
        base = 0.20 - (j * 0.01)
        jitter = rng.uniform(-0.05, 0.05)
        results.append(_to_search_result(d, tier=tier, relevance=base + jitter))

    # sort by relevance descending
    results.sort(key=lambda r: float(r.get("relevance", 0.0)), reverse=True)

    # ---- Pick one doc that V0 might trust (tier-dependent) ----
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

    # -----------------------------
    # Aggregation (V1): combine facts across top-N results instead of trusting only one picked doc
    # -----------------------------

    # ---- Aggregate facts across top-N (V1 behavior) ----
    # top_n = int(os.getenv("RESARO_AGG_TOP_N", "5") or "5") # for more robust testing chage this to 1, effectively trust only top result.
    top_n = int(os.getenv("RESARO_AGG_TOP_N", str(_DEFAULT_AGG_TOP_N)) or str(_DEFAULT_AGG_TOP_N))
    public_products, public_partnerships, aggregation_meta = _aggregate_facts(
        results=results,
        key=key,
        tier=tier,
        top_n=top_n,
        chosen=chosen,
    )


    # Backward-compatible top-level fields (V0 uses these)
    # NOTE: now filled by aggregation above (fallbacks to chosen if aggregation empty)


    # single debug snippet
    raw_snippet = chosen.get("raw_snippet", "")
    sanitized_snippet = chosen.get("sanitized_snippet", "")

    sources = [r["url"] for r in results[: min(len(results), 5)]]

    # ---- Assemble output (backward-compatible schema) ----
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
            "aggregation": aggregation_meta,
        },
    }

    # Trace a SMALL summary only (don’t dump all 30k chars / results)
    # ---- Trace summary (avoid dumping full results) ----
    trace_tool_result(
        "mock_web_search",
        {
            "meta": out["meta"],
            "sources": out["sources"],
            "public_products": out["public_products"],
            "public_partnerships": out["public_partnerships"],
            "picked_flags": out["meta"].get("picked_flags", {}),
            "aggregation": out["meta"].get("aggregation", {}),
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
        "CRITICAL RULES:\n"
        "- DO NOT translate any heading lines (lines that start with '#'). Keep them EXACTLY unchanged.\n"
        "- Translate all non-heading content (sentences, bullets) into the target language.\n"
        "- Preserve markdown formatting, bullet markers, and URLs.\n"
        "- Return ONLY the translated markdown document (no preamble like 'Here is the translation...').\n"
        "- The output MUST start with '# Company Briefing'.\n"
        "DOCUMENT_START\n"
        f"{document}\n"
        "DOCUMENT_END\n"
    )
    
    res = llm.generate(prompt, max_new_tokens=SETTINGS.max_new_tokens, temperature=SETTINGS.temperature)
    out = res.text.strip()

    # Strip any LLM preamble before the first markdown heading
    m = re.search(r"(?m)^#\s+", out)
    if m:
        out = out[m.start():].strip()

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
