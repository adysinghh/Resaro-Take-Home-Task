from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .config import SETTINGS


@dataclass
class SecurityReport:
    redacted_terms: list[str]
    injection_stripped: bool
    blocked: bool = False
    block_reason: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    tags: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)


def _vendored_secguard_src() -> Path:
    # resaro_agent/security.py -> ../secguard/src
    return Path(__file__).resolve().parents[1] / "secguard" / "src"


def _ensure_secguard_importable() -> None:
    try:
        from secguard.engine import Engine  # noqa: F401
        return
    except Exception:
        vendored_src = _vendored_secguard_src()
        if vendored_src.exists() and str(vendored_src) not in sys.path:
            sys.path.insert(0, str(vendored_src))

    from secguard.engine import Engine  # noqa: F401


@lru_cache(maxsize=1)
def _engine():
    _ensure_secguard_importable()
    from secguard.engine import Engine

    policy_path = Path(SETTINGS.secguard_policy_path)
    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path
    if not policy_path.exists():
        raise FileNotFoundError(
            f"secguard policy not found: {policy_path}. "
            "Set RESARO_SECGUARD_POLICY to a valid file path."
        )
    return Engine.from_yaml(str(policy_path))


def _promptguard_enabled() -> bool:
    val = str(getattr(SETTINGS, "secguard_enable_promptguard", False)).strip().lower()
    return val in {"1", "true", "yes", "on"}


def _apply_policy(
    text: str,
    *,
    source: str,
    stage: str,
    trust: str,
    sensitive_terms: Iterable[str] | None = None,
):
    _ensure_secguard_importable()
    from secguard.types import Context

    runtime = {
        "sensitive_terms": [str(t) for t in (sensitive_terms or []) if str(t).strip()]
    }

    # PromptGuard is enabled by default via config.
    # If explicitly disabled, set env flag so matcher short-circuits.
    added_disable_env = False
    if not _promptguard_enabled():
        has_disable_override = (
            os.getenv("SECGUARD_DISABLE_PROMPTGUARD") is not None
            or os.getenv("SECGUARD_DISABLE_DETECTORS") is not None
        )
        if not has_disable_override:
            os.environ["SECGUARD_DISABLE_PROMPTGUARD"] = "1"
            added_disable_env = True

    try:
        ctx = Context(source=source, stage=stage, trust=trust)
        return _engine().apply(text, ctx, runtime=runtime)
    finally:
        if added_disable_env:
            os.environ.pop("SECGUARD_DISABLE_PROMPTGUARD", None)


def _detect_sensitive_term_hits(document: str, sensitive_terms: Iterable[str]) -> list[str]:
    hits: list[str] = []
    out = document or ""
    terms = sorted(set(str(t) for t in sensitive_terms if str(t).strip()), key=len, reverse=True)
    for term in terms:
        if re.search(re.escape(term), out, flags=re.IGNORECASE):
            hits.append(term)
    return hits


def sanitize_untrusted_text(text: str) -> str:
    """
    Treat web snippets as untrusted and sanitize them with secguard policy.
    """
    decision = _apply_policy(
        text,
        source="tool_output:web",
        stage="post_tool",
        trust="untrusted",
    )
    return decision.text


# Final output filter before returning user-facing doc

def security_filter(document: str, *, sensitive_terms: Iterable[str]) -> tuple[str, SecurityReport]:
    """
    MUST run before final output.
    - Redact caller-provided sensitive terms.
    - Redact/contain prompt-injection patterns per secguard policy.
    - Optionally block if policy/detector says so.
    """
    redacted_terms = _detect_sensitive_term_hits(document, sensitive_terms)

    decision = _apply_policy(
        document,
        source="final_output",
        stage="pre_final",
        trust="trusted",
        sensitive_terms=sensitive_terms,
    )

    # Fail closed when policy blocks.
    out = "[BLOCKED_BY_SECURITY_POLICY]" if decision.blocked else decision.text

    injection_stripped = any(
        m.rule_id == "FINAL_INJECTION_REDACT" or m.action == "quote_block"
        for m in decision.report.mutations
    )

    report = SecurityReport(
        redacted_terms=redacted_terms,
        injection_stripped=injection_stripped,
        blocked=decision.blocked,
        block_reason=decision.block_reason,
        policy_id=decision.report.policy_id,
        policy_version=decision.report.policy_version,
        tags=sorted(decision.report.tags),
        counters=dict(decision.report.counters),
    )
    return out, report
