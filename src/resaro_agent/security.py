from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


INJECTION_PATTERNS = [
    r"(?i)\bignore (all|previous) instructions\b",
    r"(?i)\bsystem\s*:\b",
    r"(?i)\bdeveloper\s*:\b",
    r"(?i)\bassistant\s*:\b",
    r"(?i)\bexfiltrate\b",
    r"(?i)\breveal\b.*\bsecret\b",
]

@dataclass
class SecurityReport:
    redacted_terms: list[str]
    injection_stripped: bool


def sanitize_untrusted_text(text: str) -> str:
    """
    Treat web search as untrusted: remove instruction-like lines.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for ln in lines:
        if re.search(r"(?i)^\s*(system|developer|assistant|user)\s*:", ln):
            cleaned.append("[REDACTED_INJECTION_LINE]")
            continue
        if re.search(r"(?i)ignore (all|previous) instructions", ln):
            cleaned.append("[REDACTED_INJECTION_LINE]")
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def security_filter(document: str, *, sensitive_terms: Iterable[str]) -> tuple[str, SecurityReport]:
    """
    MUST run before final output.
    - Redact sensitive terms.
    - Strip common injection phrases if present.
    """
    redacted_terms: list[str] = []
    out = document

    # redact sensitive terms (exact + case-insensitive)
    for term in sorted(set([t for t in sensitive_terms if t.strip()]), key=len, reverse=True):
        if re.search(re.escape(term), out, flags=re.IGNORECASE):
            redacted_terms.append(term)
            out = re.sub(re.escape(term), "[REDACTED]", out, flags=re.IGNORECASE)

    injection_stripped = False
    for pat in INJECTION_PATTERNS:
        if re.search(pat, out):
            injection_stripped = True
            out = re.sub(pat, "[REDACTED_INJECTION]", out)

    return out, SecurityReport(redacted_terms=redacted_terms, injection_stripped=injection_stripped)
