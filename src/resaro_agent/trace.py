# src/resaro_agent/trace.py
from __future__ import annotations

import contextvars
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

TRACE_CTX: contextvars.ContextVar["TraceRecorder | None"] = contextvars.ContextVar("RESARO_TRACE", default=None)

def now_ms() -> int:
    return int(time.time() * 1000)

def _short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

def _clip(s: str, n: int = 800) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[:n] + "…")

@dataclass
class TraceEvent:
    ts_ms: int
    kind: str
    data: Dict[str, Any]

class TraceRecorder:
    """
    Stores events in-memory and optionally writes JSONL incrementally.
    Good for eval: you can compare traces across V0/V1/V2/V3.
    """
    def __init__(self, run_id: str, out_dir: str, *, stream_jsonl: bool = True):
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stream_jsonl = stream_jsonl
        self.events: List[TraceEvent] = []
        self._jsonl_path = self.out_dir / f"{run_id}_trace.jsonl"
        if self.stream_jsonl:
            self._jsonl_path.write_text("", encoding="utf-8")

    def add(self, kind: str, **data: Any) -> None:
        ev = TraceEvent(ts_ms=now_ms(), kind=kind, data=data)
        self.events.append(ev)
        if self.stream_jsonl:
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

    def dump_json(self) -> Path:
        p = self.out_dir / f"{self.run_id}_trace.json"
        p.write_text(json.dumps([asdict(e) for e in self.events], indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def dump_md(self) -> Path:
        p = self.out_dir / f"{self.run_id}_trace.md"
        lines: List[str] = []
        lines.append(f"# Agent Trace: {self.run_id}\n")
        for i, e in enumerate(self.events, 1):
            lines.append(f"## {i}. {e.kind}")
            lines.append(f"- **ts_ms:** {e.ts_ms}")
            for k, v in e.data.items():
                if isinstance(v, str):
                    lines.append(f"- **{k}:** {_clip(v, 1200)}")
                else:
                    vv = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    lines.append(f"- **{k}:** {_clip(vv, 1200)}")
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

def get_trace() -> Optional[TraceRecorder]:
    return TRACE_CTX.get()

def set_trace(tr: Optional[TraceRecorder]):
    return TRACE_CTX.set(tr)

def reset_trace(token):
    TRACE_CTX.reset(token)

# Convenience: safe-ish logging helpers
def trace_llm(kind: str, prompt: str, completion: str, model: str, latency_ms: int, tokens_est: int):
    tr = get_trace()
    if not tr:
        return
    tr.add(
        kind,
        model=model,
        latency_ms=latency_ms,
        tokens_est=tokens_est,
        prompt_hash=_short_hash(prompt),
        prompt_preview=_clip(prompt, 1200),
        completion_preview=_clip(completion, 1200),
        completion_raw=completion,  # keep full by default; you can gate via env if you want
    )

def trace_tool_call(tool: str, args: Dict[str, Any]):
    tr = get_trace()
    if tr:
        tr.add("tool_call", tool=tool, args=args)

def trace_tool_result(tool: str, result: Any, *, preview_clip: int = 1200, extra: Optional[Dict[str, Any]] = None):
    tr = get_trace()
    if not tr:
        return
    payload = {"tool": tool, "result_preview": _clip(result, preview_clip)}
    if extra:
        payload.update(extra)
    tr.add("tool_result", **payload)
