from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


REQUIRED_HEADINGS = [
    "# Company Briefing",
    "## Overview",
    "## Products",
    "## Partnerships",
    "## Risk Notes",
    "## Sources",
]

LANG_HINTS = {
    "english": ["the", "and", "with", "company"],
    "german": ["und", "der", "die", "ist", "eine"],
    "french": ["et", "le", "la", "est", "une"],
    "spanish": ["y", "el", "la", "es", "una"],
}


@dataclass
class ValidationResult:
    ok: bool
    template_coverage: float
    missing_headings: list[str]
    leakage_found: bool
    language_ok: bool
    tool_requirements_ok: bool
    reasons: list[str]


def check_template(document: str) -> tuple[float, list[str]]:
    missing = [h for h in REQUIRED_HEADINGS if h not in document]
    coverage = (len(REQUIRED_HEADINGS) - len(missing)) / max(1, len(REQUIRED_HEADINGS))
    return coverage, missing


def check_leakage(document: str, sensitive_terms: Iterable[str]) -> bool:
    for term in sensitive_terms:
        if term.strip() and re.search(re.escape(term), document, flags=re.IGNORECASE):
            return True
    return False


def check_language(document: str, target_language: str) -> bool:
    lang = target_language.strip().lower()
    if lang not in LANG_HINTS:
        # unknown language: don't hard fail in V0
        return True
    hints = LANG_HINTS[lang]
    doc_low = document.lower()
    hits = sum(1 for w in hints if re.search(rf"\b{re.escape(w)}\b", doc_low))
    return hits >= max(1, len(hints) // 3)  # lenient heuristic


def validate_document(
    *,
    document: str,
    target_language: str,
    sensitive_terms: Iterable[str],
    tool_log: list[dict],
    needs_translation: bool,
) -> ValidationResult:
    reasons: list[str] = []

    coverage, missing = check_template(document)
    if missing:
        reasons.append(f"Missing headings: {missing}")

    leakage = check_leakage(document, sensitive_terms)
    if leakage:
        reasons.append("Leakage detected: sensitive term present in document")

    lang_ok = check_language(document, target_language)
    if not lang_ok:
        reasons.append(f"Language check failed for target_language={target_language}")

    # Tool requirements: must use required tools (and translate only if needed)
    called = [e.get("tool") for e in tool_log]
    
    # NOTE: security_filter runs AFTER validate in the graph; enforce it in finalize, not here.
    must = ["get_company_info", "mock_web_search", "generate_document"]
    tool_ok = all(m in called for m in must)

    # If translation is needed, ensure it ran at least once.
    if needs_translation:
        tool_ok = tool_ok and ("translate_document" in called)

    if not tool_ok:
        reasons.append(f"Tool requirements not satisfied. Called={called}")

    ok = (coverage == 1.0) and (not leakage) and lang_ok and tool_ok
    return ValidationResult(
        ok=ok,
        template_coverage=coverage,
        missing_headings=missing,
        leakage_found=leakage,
        language_ok=lang_ok,
        tool_requirements_ok=tool_ok,
        reasons=reasons,
    )
