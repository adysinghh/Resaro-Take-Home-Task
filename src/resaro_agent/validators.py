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


HEADING_PATTERNS = {
    "# Company Briefing": r"(?m)^#\s*Company\s+Briefing\b.*$",
    "## Overview": r"(?m)^##\s*Overview\s*$",
    "## Products": r"(?m)^##\s*Products\s*$",
    "## Partnerships": r"(?m)^##\s*Partnerships\s*$",
    "## Risk Notes": r"(?m)^##\s*Risk\s+Notes\s*$",
    "## Sources": r"(?m)^##\s*Sources\s*$",
}

# Compile once (non-breaking, just avoids re-parsing patterns repeatedly).
_COMPILED_HEADING_PATTERNS = {k: re.compile(v) for k, v in HEADING_PATTERNS.items()}


def check_template(document: str) -> tuple[float, list[str]]:
    """
    Checks whether the document matches the required markdown template.

    Returns:
        (coverage, missing_headings)
    """
    doc = document or ""
    missing: list[str] = []

    for heading in REQUIRED_HEADINGS:
        compiled = _COMPILED_HEADING_PATTERNS.get(heading)
        if compiled is None:
            # Fallback: if a heading doesn't have a regex pattern, do a substring check.
            # (Preserves legacy behavior.)
            if heading not in doc:
                missing.append(heading)
            continue

        if not compiled.search(doc):
            missing.append(heading)

    total = max(1, len(REQUIRED_HEADINGS))
    coverage = (len(REQUIRED_HEADINGS) - len(missing)) / total
    return coverage, missing


def check_leakage(document: str, sensitive_terms: Iterable[str]) -> bool:
    """
    Returns True if any sensitive term appears in the document (case-insensitive).
    """
    doc = document or ""
    for term in sensitive_terms:
        t = term.strip()
        if not t:
            continue
        if re.search(re.escape(t), doc, flags=re.IGNORECASE):
            return True
    return False


def _language_body(document: str) -> str:
    """
    Extracts "prose-like" content from the markdown document for a simple language heuristic.

    Keeps the logic intentionally conservative:
    - Skips headings, list items, and URL lines (sources).
    - Joins remaining non-empty lines into one string.
    """
    if not document:
        return ""

    lines: list[str] = []
    for ln in document.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):  # headings
            continue
        if s.startswith("- "):  # list items (often entity names)
            continue
        if "http://" in s or "https://" in s:  # sources
            continue
        lines.append(s)

    return " ".join(lines)


def check_language(document: str, target_language: str) -> bool:
    """
    Lightweight heuristic: checks for a couple of common function words in the target language.
    """
    lang = target_language.strip().lower()
    if lang not in LANG_HINTS:
        return True  # unknown language: don't hard fail

    body = _language_body(document).lower()

    # If there's basically no prose, fail (prevents false passes).
    if len(body) < 30:
        return False

    hints = LANG_HINTS[lang]
    hits = sum(1 for w in hints if re.search(rf"\b{re.escape(w)}\b", body))

    # Lenient: 1 hit is enough for non-English, 2 for English.
    threshold = 2 if lang == "english" else 1
    return hits >= threshold


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

    # Language check only when translation is required.
    # (Existing behavior: enforce only when needs_translation is True.)
    lang_ok = True
    if needs_translation:
        lang_ok = check_language(document, target_language)
        if not lang_ok:
            reasons.append(f"Language check failed for target_language={target_language}")

    # Tool requirements: must use required tools (and translate only if needed).
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
