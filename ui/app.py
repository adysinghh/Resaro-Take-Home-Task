import os
import time
import html
import json
import base64
import subprocess
import re
import ast
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SCRIPT = REPO_ROOT / "scripts" / "bootstrap.py"

# ✅ hardcode your local paths here
DEMO_VIDEO_PATH = Path("/ABS/PATH/TO/demo.mp4")

# optional
DOC_VIDEO_PATH = Path("/ABS/PATH/TO/doc_video.mp4")   # optional
DOC_DECK_PDF_PATH = Path("/ABS/PATH/TO/doc_deck.pdf") # optional

# Data scripts
SCRIPT_GEN_DB = REPO_ROOT / "scripts" / "generate_synth_data.py"
SCRIPT_PIN = REPO_ROOT / "scripts" / "pin_canonical_companies.py"
SCRIPT_BUILD_CORPUS = REPO_ROOT / "scripts" / "build_web_corpus.py"
SCRIPT_INSPECT = REPO_ROOT / "scripts" / "inspect_hardsim.py"

DB_PATH = REPO_ROOT / "src" / "data" / "company_db.json"
CORPUS_PATH = REPO_ROOT / "src" / "data" / "web_corpus.jsonl"


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Resaro Agentic Eval Demo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Session state
# ============================================================
def _init_state():
    defaults = {
        "busy": False,
        "setup_done": False,

        # eval logs
        "logs": [],

        # eval trace state
        "trace_offset": 0,
        "trace_file": None,

        # data pipeline
        "data_busy": False,
        "data_logs": [],
        "data_summary": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ============================================================
# Styling (premium dark + gold)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg0: #07110E;
  --bg1: #0B1A16;
  --card: rgba(14, 30, 26, 0.72);
  --stroke: rgba(200, 178, 115, 0.22);
  --stroke2: rgba(244, 241, 234, 0.10);
  --gold: #C8B273;
  --text: #F4F1EA;
  --muted: rgba(244, 241, 234, 0.72);
  --good: #7CFFB2;
  --warn: #FFD37C;
  --bad:  #FF7C7C;
}

html, body, [class*="css"]  {
  font-family: "Space Grotesk", system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}

.block-container {
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}

.hero {
  padding: 18px 18px 14px 18px;
  border-radius: 18px;
  background: radial-gradient(1200px 400px at 20% 0%, rgba(200,178,115,0.18), transparent 60%),
              radial-gradient(900px 360px at 80% 10%, rgba(124,255,178,0.08), transparent 55%),
              linear-gradient(180deg, rgba(11,26,22,0.95), rgba(7,17,14,0.95));
  border: 1px solid var(--stroke2);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.3px;
  background: rgba(200,178,115,0.06);
}

.h-title {
  margin-top: 8px;
  font-size: 34px;
  line-height: 1.1;
  color: var(--text);
  font-weight: 600;
  letter-spacing: 0.2px;
}

.h-sub {
  margin-top: 6px;
  color: var(--muted);
  font-size: 14px;
}

.card {
  border-radius: 18px;
  background: var(--card);
  border: 1px solid var(--stroke2);
  padding: 14px 14px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

.terminal {
  border-radius: 18px;
  background: rgba(5, 12, 10, 0.86);
  border: 1px solid rgba(200,178,115,0.18);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
  padding: 12px 12px;
  overflow: auto;
}

.terminal pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
  font-size: 12px;
  line-height: 1.35;
  color: rgba(244,241,234,0.88);
}

.termline-info { color: rgba(244,241,234,0.88); }
.termline-ok   { color: var(--good); }
.termline-warn { color: var(--warn); }
.termline-bad  { color: var(--bad); }

.smallhint {
  color: var(--muted);
  font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers: logs + terminal rendering
# ============================================================
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log_eval(msg: str, level: str = "info"):
    lvl = level.lower()
    if lvl not in {"info", "ok", "warn", "bad"}:
        lvl = "info"
    st.session_state.logs.append((_ts(), msg, lvl))
    st.session_state.logs = st.session_state.logs[-800:]

def log_data(msg: str, level: str = "info"):
    lvl = level.lower()
    if lvl not in {"info", "ok", "warn", "bad"}:
        lvl = "info"
    st.session_state.data_logs.append((_ts(), msg, lvl))
    st.session_state.data_logs = st.session_state.data_logs[-1200:]

def render_terminal(container, logs: List[Tuple[str, str, str]], height_px: int = 420):
    lines = []
    for ts, msg, lvl in logs:
        cls = {
            "info": "termline-info",
            "ok": "termline-ok",
            "warn": "termline-warn",
            "bad": "termline-bad",
        }[lvl]
        safe = html.escape(f"[{ts}] {msg}")
        lines.append(f"<span class='{cls}'>{safe}</span>")
    blob = "<br/>".join(lines) if lines else "<span class='termline-info'>[--:--:--] waiting...</span>"
    container.markdown(
        f"<div class='terminal' style='height:{height_px}px;'><pre>{blob}</pre></div>",
        unsafe_allow_html=True
    )


# ============================================================
# Helpers: subprocess streaming (unbuffered)
# ============================================================
def run_subprocess_stream(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None):
    env2 = dict(os.environ)
    if env:
        env2.update(env)

    # IMPORTANT: ensure src/ imports work
    root_str = str(REPO_ROOT)
    env2["PYTHONPATH"] = (env2.get("PYTHONPATH", "") + (":" if env2.get("PYTHONPATH") else "") + root_str)

    # unbuffered python
    env2["PYTHONUNBUFFERED"] = "1"

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        yield line.rstrip("\n")
    yield f"[exit_code={p.wait()}]"


# ============================================================
# Helpers: file + stats
# ============================================================
PINNED = [
    "Asteron", "Novacrest", "HelioForge", "Quartzline", "Boreal Dynamics",
    "MiraWorks", "Syntera Labs", "CobaltWave", "Pinebridge Systems", "OrchidShield",
]

def read_company_db_summary() -> Dict[str, Any]:
    if not DB_PATH.exists():
        return {"ok": False, "error": f"missing {DB_PATH}"}
    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    companies = db.get("companies", [])
    first10 = [c.get("name") for c in companies[:10]]
    pinned_ok = (first10 == PINNED)
    return {
        "ok": True,
        "companies": len(companies),
        "first10": first10,
        "pinned_ok": pinned_ok,
    }

def corpus_flag_stats(corpus_path: Path) -> Dict[str, Any]:
    if not corpus_path.exists():
        return {"ok": False, "error": f"missing {corpus_path}"}

    n = 0
    cnt = {"injected": 0, "contradictory": 0, "stale": 0, "corrupted_schema": 0}

    # 30k lines is fine to scan
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fl = rec.get("flags", {}) or {}
            for k in cnt.keys():
                if fl.get(k):
                    cnt[k] += 1

    rates = {k: (cnt[k] / n if n else 0.0) for k in cnt.keys()}
    return {
        "ok": True,
        "docs": n,
        "counts": cnt,
        "rates": rates,
    }

def parse_inspect_hardsim_output(text: str) -> Dict[str, Any]:
    """
    Parse the printed meta dicts from scripts/inspect_hardsim.py output.
    We extract: tier -> {k, noise_share, aggregation.used_n, picked_flags}
    """
    out = {}
    # find blocks like:
    # TIER=EASY  company=Asteron
    # meta: {...}
    tier_pat = re.compile(r"TIER=(EASY|REALISTIC|HARD)\s+company=.*\nmeta:\s*(\{.*\})", re.IGNORECASE)
    for m in tier_pat.finditer(text):
        tier = m.group(1).lower()
        meta_str = m.group(2)
        try:
            meta = ast.literal_eval(meta_str)
        except Exception:
            meta = {"_raw": meta_str}

        agg = meta.get("aggregation", {}) if isinstance(meta, dict) else {}
        picked_flags = meta.get("picked_flags", {}) if isinstance(meta, dict) else {}
        out[tier] = {
            "k": meta.get("k"),
            "noise_share": meta.get("noise_share"),
            "picked_flags": picked_flags,
            "aggregation": {
                "enabled": agg.get("enabled"),
                "top_n": agg.get("top_n"),
                "used_n": agg.get("used_n"),
                "used_indexes": agg.get("used_indexes"),
            }
        }
    return out


# ============================================================
# Existing: eval trace helpers (kept)
# ============================================================
def newest_file(glob_pattern: str) -> Optional[Path]:
    files = list(REPO_ROOT.glob(glob_pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def tail_text_file(path: Path, max_lines: int = 120) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(txt[-max_lines:])
    except Exception as e:
        return f"[tail error] {type(e).__name__}: {e}"

def tail_jsonl_file(path: Path, offset: int, max_new_lines: int = 200):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(offset)
            lines = []
            for _ in range(max_new_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
            new_offset = f.tell()

        events = []
        for ln in lines:
            if not ln.strip():
                continue
            try:
                events.append(json.loads(ln))
            except Exception:
                events.append({"_raw": ln})

        raw_tail = "\n".join(lines[-50:])
        return events, new_offset, raw_tail
    except Exception as e:
        return [], offset, f"[jsonl tail error] {type(e).__name__}: {e}"

def count_run_logs():
    tiers = ["easy", "realistic", "hard"]
    out = {}
    for t in tiers:
        d = REPO_ROOT / "reports" / f"run_logs_{t}"
        out[t] = len(list(d.glob("task_*.json"))) if d.exists() else 0
    return out


# ============================================================
# Setup pipeline (existing)
# ============================================================
def setup_pipeline(progress, status_ph, terminal_ph):
    st.session_state.busy = True
    try:
        log_eval("setup: starting bootstrap", "info")

        if not BOOTSTRAP_SCRIPT.exists():
            log_eval(f"bootstrap missing: {BOOTSTRAP_SCRIPT}", "bad")
            raise FileNotFoundError(f"Missing {BOOTSTRAP_SCRIPT}")

        status_ph.markdown("**Installing requirements (local-only)…**")
        progress.progress(10)

        cmd = [os.environ.get("PYTHON", "python"), str(BOOTSTRAP_SCRIPT)]
        log_eval(f"running: {' '.join(cmd)}", "info")

        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            lvl = "info"
            if "Traceback" in out or "ERROR" in out:
                lvl = "bad"
            elif "[exit_code=0]" in out or "done ✅" in out:
                lvl = "ok"
            log_eval(out, lvl)
            render_terminal(terminal_ph, st.session_state.logs, height_px=420)

        progress.progress(100)
        status_ph.markdown("**Setup complete ✅**")
        st.session_state.setup_done = True
        log_eval("setup_done=true", "ok")

    except Exception as e:
        status_ph.markdown("**Setup failed ❌**")
        log_eval(f"setup_error: {type(e).__name__}: {e}", "bad")
    finally:
        st.session_state.busy = False


# ============================================================
# Eval runner (existing)
# ============================================================
def run_eval(n_tasks: int, max_redteam: int, terminal_ph,
             progress_meta_ph, trace_json_ph, trace_tail_ph, brain_tail_ph):
    st.session_state.busy = True
    try:
        cmd_str = (
            f"RESARO_EVAL_N_TASKS={n_tasks} "
            f"RESARO_EVAL_MAX_REDTEAM={max_redteam} "
            f"PYTHONPATH=. python -u scripts/run_eval.py"
        )
        log_eval(f"running: {cmd_str}", "info")
        cmd = ["bash", "-lc", cmd_str]

        st.session_state.trace_offset = 0
        st.session_state.trace_file = None

        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            lvl = "info"
            if "Traceback" in out or "ERROR" in out:
                lvl = "bad"
            elif "=== V0 EVAL SUMMARY" in out:
                lvl = "ok"
            log_eval(out, lvl)
            render_terminal(terminal_ph, st.session_state.logs, height_px=420)

            counts = count_run_logs()
            progress_meta_ph.markdown(
                f"**run_logs created:** easy={counts['easy']} • realistic={counts['realistic']} • hard={counts['hard']}"
            )

            latest_trace = newest_file("reports/traces_fast/*_TRACE.jsonl")
            latest_brain = newest_file("reports/traces_fast/*_BRAIN_LOG.md")

            if latest_trace and str(latest_trace) != st.session_state.trace_file:
                st.session_state.trace_file = str(latest_trace)
                st.session_state.trace_offset = 0
                log_eval(f"trace_file: {latest_trace.name}", "ok")

            if st.session_state.trace_file:
                tf = Path(st.session_state.trace_file)
                events, new_off, raw_tail = tail_jsonl_file(tf, st.session_state.trace_offset)
                st.session_state.trace_offset = new_off
                if events:
                    trace_json_ph.json(events[-1])
                trace_tail_ph.code(raw_tail or "(waiting for trace events...)")
            else:
                trace_json_ph.info("Waiting for TRACE.jsonl to appear...")
                trace_tail_ph.code("")

            if latest_brain:
                brain_tail_ph.code(tail_text_file(latest_brain, max_lines=120))
            else:
                brain_tail_ph.info("Waiting for BRAIN_LOG.md to appear...")

            time.sleep(0.02)

    finally:
        st.session_state.busy = False


# ============================================================
# NEW: One-click Synthetic Data Pipeline (UI automation)
# ============================================================
def run_data_pipeline(pages_per_company: int, seed: int, progress_ph, status_ph, console_ph, summary_ph):
    """
    Runs:
      1) generate_synth_data.py
      2) pin_canonical_companies.py
      3) build_web_corpus.py --pages_per_company N --seed S
      4) verify db + corpus stats (in-process)
      5) inspect_hardsim.py (and parse noise/k summary)
    Streams logs live to console.
    """
    st.session_state.data_busy = True
    st.session_state.data_logs = []
    st.session_state.data_summary = None

    def step(pct: int, label: str):
        progress_ph.progress(pct)
        status_ph.markdown(f"**{label}**")

    try:
        # ---- sanity checks
        required = [SCRIPT_GEN_DB, SCRIPT_PIN, SCRIPT_BUILD_CORPUS, SCRIPT_INSPECT]
        missing = [p for p in required if not p.exists()]
        if missing:
            for m in missing:
                log_data(f"missing script: {m}", "bad")
            render_terminal(console_ph, st.session_state.data_logs, height_px=320)
            raise FileNotFoundError(f"Missing scripts: {[str(x) for x in missing]}")

        # ---- Step 1
        step(5, "Step 1/5 — Generating synthetic company DB")
        cmd = [os.environ.get("PYTHON", "python"), "-u", str(SCRIPT_GEN_DB)]
        log_data(f"running: {' '.join(cmd)}", "info")
        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            log_data(out, "info" if "Traceback" not in out else "bad")
            render_terminal(console_ph, st.session_state.data_logs, height_px=320)

        # ---- Step 2
        step(25, "Step 2/5 — Pinning canonical 10 names")
        cmd = [os.environ.get("PYTHON", "python"), "-u", str(SCRIPT_PIN)]
        log_data(f"running: {' '.join(cmd)}", "info")
        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            log_data(out, "info" if "Traceback" not in out else "bad")
            render_terminal(console_ph, st.session_state.data_logs, height_px=320)

        # ---- Verify DB right away (print summary)
        db_sum = read_company_db_summary()
        if not db_sum.get("ok"):
            log_data(f"DB verify failed: {db_sum.get('error')}", "bad")
        else:
            log_data(f"DB companies: {db_sum['companies']}", "ok")
            log_data(f"DB first10: {db_sum['first10']}", "info")
            log_data(f"Canonical pinned OK? {db_sum['pinned_ok']}", "ok" if db_sum["pinned_ok"] else "bad")

        render_terminal(console_ph, st.session_state.data_logs, height_px=320)

        # ---- Step 3
        step(45, "Step 3/5 — Building synthetic web corpus")
        expected_docs = 1000 * pages_per_company
        cmd = [
            os.environ.get("PYTHON", "python"), "-u", str(SCRIPT_BUILD_CORPUS),
            "--db", "src/data/company_db.json",
            "--out", "src/data/web_corpus.jsonl",
            "--pages_per_company", str(pages_per_company),
            "--seed", str(seed),
        ]
        log_data(f"running: {' '.join(cmd)}", "info")
        log_data(f"expected corpus docs: {expected_docs}", "info")

        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            log_data(out, "info" if "Traceback" not in out else "bad")
            render_terminal(console_ph, st.session_state.data_logs, height_px=320)

        # ---- Step 4: corpus stats
        step(70, "Step 4/5 — Verifying corpus size + noise flags")
        corp_sum = corpus_flag_stats(CORPUS_PATH)
        if not corp_sum.get("ok"):
            log_data(f"Corpus verify failed: {corp_sum.get('error')}", "bad")
        else:
            docs = corp_sum["docs"]
            ok_size = (docs == expected_docs)
            log_data(f"Corpus docs: {docs} (expected {expected_docs})", "ok" if ok_size else "warn")
            for k, r in corp_sum["rates"].items():
                log_data(f"flag_rate[{k}]: {r:.3f}", "info")

        render_terminal(console_ph, st.session_state.data_logs, height_px=320)

        # ---- Step 5: inspect tiers
        step(85, "Step 5/5 — Inspecting HardSim tiers (easy/realistic/hard)")
        cmd = [os.environ.get("PYTHON", "python"), "-u", str(SCRIPT_INSPECT)]
        log_data(f"running: {' '.join(cmd)}", "info")

        inspect_lines = []
        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            inspect_lines.append(out)
            log_data(out, "info" if "Traceback" not in out else "bad")
            render_terminal(console_ph, st.session_state.data_logs, height_px=320)

        inspect_text = "\n".join(inspect_lines)
        tier_summary = parse_inspect_hardsim_output(inspect_text)
        if tier_summary:
            log_data("Parsed tier noise summary:", "ok")
            for tier, meta in tier_summary.items():
                log_data(f"{tier}: k={meta.get('k')} noise_share={meta.get('noise_share')} used_n={meta.get('aggregation',{}).get('used_n')}", "info")
        else:
            log_data("Could not parse tier summary from inspect_hardsim output (still OK).", "warn")

        # ---- final summary object
        final_summary = {
            "company_db": db_sum,
            "corpus": {
                "pages_per_company": pages_per_company,
                "seed": seed,
                "expected_docs": expected_docs,
                **corp_sum,
            },
            "tiers": tier_summary,
            "paths": {
                "company_db": str(DB_PATH),
                "web_corpus": str(CORPUS_PATH),
            }
        }
        st.session_state.data_summary = final_summary

        # show summary in UI
        summary_ph.json(final_summary)

        step(100, "Synthetic Data Pipeline complete ✅")
        log_data("PIPELINE COMPLETE ✅", "ok")
        render_terminal(console_ph, st.session_state.data_logs, height_px=320)

    except Exception as e:
        status_ph.markdown("**Synthetic Data Pipeline failed ❌**")
        log_data(f"pipeline_error: {type(e).__name__}: {e}", "bad")
        render_terminal(console_ph, st.session_state.data_logs, height_px=320)
    finally:
        st.session_state.data_busy = False


# ============================================================
# PDF helper (existing)
# ============================================================
def render_pdf_inline_from_path(pdf_path, height: int = 560):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        st.error(f"PDF not found: {pdf_path}")
        return

    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    html_blob = f"""
    <div style="
        border: 1px solid rgba(200,178,115,0.18);
        border-radius: 14px;
        overflow: hidden;
        background: rgba(0,0,0,0.15);
    ">
      <iframe
        src="data:application/pdf;base64,{b64}"
        width="100%"
        height="{height}"
        style="border:0;"
      ></iframe>
    </div>
    """
    components.html(html_blob, height=height + 30, scrolling=False)


def render_pdf_as_images(pdf_path, max_pages: int = 20, zoom: float = 1.4):
    import fitz  # PyMuPDF
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        st.error(f"PDF not found: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    n = min(len(doc), max_pages)
    for i in range(n):
        page = doc[i]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        st.image(pix.tobytes("png"), use_container_width=True)

    if len(doc) > max_pages:
        st.caption(f"Showing first {max_pages} pages (total: {len(doc)}).")


# ============================================================
# Hero
# ============================================================
st.markdown(
    """
<div class="hero">
  <span class="badge">Agentic Application Testing • UI Demo</span>
  <div class="h-title">Resaro Take-Home • Live Eval Console</div>
  <div class="h-sub">
    One-click setup • One-click synthetic data pipeline • Run evals • Stream logs • Browse artifacts
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# ============================================================
# Layout
# ============================================================
left, right = st.columns([0.50, 0.50], gap="large")


# ============================================================
# LEFT COLUMN
# ============================================================
with left:
    # -----------------------------
    # One-Click Setup
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 One-Click Setup (Local)")
    st.caption("Installs deps via bootstrap. (Uses your local python + pip.)")

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        setup_toggle = st.toggle(
            "Setup project",
            value=st.session_state.setup_done,
            disabled=st.session_state.busy or st.session_state.data_busy,
            key="setup_toggle",
        )
    with c2:
        st.markdown(
            f"<div class='smallhint'>status: <b>{'READY' if st.session_state.setup_done else 'NOT READY'}</b></div>",
            unsafe_allow_html=True,
        )

    setup_progress = st.progress(0)
    setup_status = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # -----------------------------
    # NEW: Synthetic Data Pipeline (One-click)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧬 Synthetic Data Pipeline (One-click)")
    st.caption("Runs DB gen → pin canonical names → build web corpus → verify corpus flags → inspect tiers (noise).")

    pcol1, pcol2 = st.columns([0.5, 0.5])
    with pcol1:
        pages_per_company = st.number_input(
            "pages_per_company",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            disabled=st.session_state.data_busy or st.session_state.busy,
            help="30 pages/company × 1000 companies = 30k corpus.",
        )
    with pcol2:
        seed = st.number_input(
            "seed",
            min_value=0,
            max_value=10_000,
            value=7,
            step=1,
            disabled=st.session_state.data_busy or st.session_state.busy,
        )

    dbtn1, dbtn2, dbtn3 = st.columns([0.42, 0.28, 0.30])
    with dbtn1:
        run_data_btn = st.button(
            "Run data pipeline",
            use_container_width=True,
            disabled=st.session_state.data_busy or st.session_state.busy,
        )
    with dbtn2:
        verify_only_btn = st.button(
            "Verify only",
            use_container_width=True,
            disabled=st.session_state.data_busy or st.session_state.busy,
            help="Does NOT regenerate. Just reads existing DB/corpus + runs inspect_hardsim.",
        )
    with dbtn3:
        clear_data_console = st.button(
            "Clear console",
            use_container_width=True,
            disabled=st.session_state.data_busy,
        )

    data_progress = st.progress(0)
    data_status = st.empty()

    st.markdown("**Summary (auto):**")
    data_summary_ph = st.empty()

    st.markdown("**Console:**")
    data_console_ph = st.empty()
    render_terminal(data_console_ph, st.session_state.data_logs, height_px=320)

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # -----------------------------
    # Run Simulation (Real)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ▶ Run Simulation (Real)")
    st.caption("Runs scripts/run_eval.py with env vars and streams stdout + trace tails.")

    n_tasks = st.slider(
        "RESARO_EVAL_N_TASKS",
        1, 200, 50, 1,
        disabled=st.session_state.busy or st.session_state.data_busy,
        key="sim_n_tasks",
    )
    max_redteam = st.slider(
        "RESARO_EVAL_MAX_REDTEAM",
        0, 10, 1, 1,
        disabled=st.session_state.busy or st.session_state.data_busy,
        key="sim_max_redteam",
    )

    b1, b2 = st.columns([0.55, 0.45], gap="medium")
    with b1:
        run_sim = st.button(
            "Run simulation",
            use_container_width=True,
            disabled=st.session_state.busy or st.session_state.data_busy,
            key="sim_run",
        )
    with b2:
        quick_run = st.button(
            "⚡ Quick run (3 tasks)",
            use_container_width=True,
            disabled=st.session_state.busy or st.session_state.data_busy,
            key="sim_quick",
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # -----------------------------
    # Trace card (eval)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Current Run Trace (Live)")
    st.caption("Shows latest TRACE.jsonl + BRAIN_LOG.md as they are written (if enabled in your agent).")

    progress_meta_ph = st.empty()
    trace_json_ph = st.empty()
    trace_tail_ph = st.empty()
    brain_tail_ph = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # -----------------------------
    # Terminal (eval)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Live Logs (Eval)")
    st.caption("Streaming stdout from run_eval.py.")
    eval_terminal_ph = st.empty()
    render_terminal(eval_terminal_ph, st.session_state.logs, height_px=420)
    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # Handlers (must be after placeholders exist)
    # ============================================================
    if clear_data_console and not st.session_state.data_busy:
        st.session_state.data_logs = []
        st.session_state.data_summary = None
        data_summary_ph.empty()
        render_terminal(data_console_ph, st.session_state.data_logs, height_px=320)

    if setup_toggle and not st.session_state.setup_done and not st.session_state.busy and not st.session_state.data_busy:
        setup_status.markdown("**Starting setup…**")
        setup_pipeline(setup_progress, setup_status, eval_terminal_ph)

    if run_data_btn and not st.session_state.data_busy and not st.session_state.busy:
        run_data_pipeline(
            pages_per_company=int(pages_per_company),
            seed=int(seed),
            progress_ph=data_progress,
            status_ph=data_status,
            console_ph=data_console_ph,
            summary_ph=data_summary_ph,
        )

    if verify_only_btn and not st.session_state.data_busy and not st.session_state.busy:
        # "verify only": no regeneration, but still show progress + console
        st.session_state.data_logs = []
        data_progress.progress(0)
        data_status.markdown("**Verifying existing DB + corpus…**")
        log_data("VERIFY ONLY: reading company_db.json + web_corpus.jsonl + running inspect_hardsim.py", "info")
        render_terminal(data_console_ph, st.session_state.data_logs, height_px=320)

        # DB
        db_sum = read_company_db_summary()
        if db_sum.get("ok"):
            log_data(f"DB companies: {db_sum['companies']}", "ok")
            log_data(f"DB first10: {db_sum['first10']}", "info")
            log_data(f"Canonical pinned OK? {db_sum['pinned_ok']}", "ok" if db_sum["pinned_ok"] else "bad")
        else:
            log_data(f"DB verify failed: {db_sum.get('error')}", "bad")

        data_progress.progress(35)
        render_terminal(data_console_ph, st.session_state.data_logs, height_px=320)

        # Corpus stats
        corp_sum = corpus_flag_stats(CORPUS_PATH)
        if corp_sum.get("ok"):
            log_data(f"Corpus docs: {corp_sum['docs']}", "ok")
            for k, r in corp_sum["rates"].items():
                log_data(f"flag_rate[{k}]: {r:.3f}", "info")
        else:
            log_data(f"Corpus verify failed: {corp_sum.get('error')}", "bad")

        data_progress.progress(70)
        render_terminal(data_console_ph, st.session_state.data_logs, height_px=320)

        # Inspect tiers
        inspect_lines = []
        cmd = [os.environ.get("PYTHON", "python"), "-u", str(SCRIPT_INSPECT)]
        log_data(f"running: {' '.join(cmd)}", "info")
        for out in run_subprocess_stream(cmd, cwd=REPO_ROOT):
            inspect_lines.append(out)
            log_data(out, "info" if "Traceback" not in out else "bad")
            render_terminal(data_console_ph, st.session_state.data_logs, height_px=320)

        tier_summary = parse_inspect_hardsim_output("\n".join(inspect_lines))
        final_summary = {
            "company_db": db_sum,
            "corpus": corp_sum,
            "tiers": tier_summary,
            "paths": {"company_db": str(DB_PATH), "web_corpus": str(CORPUS_PATH)},
        }
        st.session_state.data_summary = final_summary
        data_summary_ph.json(final_summary)

        data_progress.progress(100)
        data_status.markdown("**Verification complete ✅**")

    if run_sim and not st.session_state.busy and not st.session_state.data_busy:
        run_eval(int(n_tasks), int(max_redteam), eval_terminal_ph, progress_meta_ph, trace_json_ph, trace_tail_ph, brain_tail_ph)

    if quick_run and not st.session_state.busy and not st.session_state.data_busy:
        run_eval(3, 1, eval_terminal_ph, progress_meta_ph, trace_json_ph, trace_tail_ph, brain_tail_ph)


# ============================================================
# RIGHT COLUMN
# ============================================================
with right:
    # -----------------------------
    # Demo assets (as you had)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎥 Demo (Video)")
    st.caption("Hardcoded paths.")

    vcol, scol = st.columns([0.52, 0.48], gap="medium")

    with vcol:
        if DEMO_VIDEO_PATH.exists():
            st.video(DEMO_VIDEO_PATH.read_bytes())
        else:
            st.warning(f"Video not found: {DEMO_VIDEO_PATH}")

        if DOC_VIDEO_PATH.exists():
            st.write("")
            st.caption("Doc video")
            st.video(DOC_VIDEO_PATH.read_bytes())

    with scol:

        if DOC_DECK_PDF_PATH.exists():
            st.write("")
            st.caption("Doc deck (PDF)")
            render_pdf_inline_from_path(DOC_DECK_PDF_PATH, height=420)

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # -----------------------------
    # NEW: Artifacts explorer (run_logs)
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📄 Run Artifacts Explorer")
    st.caption("Browse generated .md / .json outputs in reports/run_logs_{tier} and view below.")

    tier = st.selectbox(
        "Select tier",
        ["easy", "realistic", "hard"],
        index=0,
        key="artifact_tier",
    )
    run_dir = REPO_ROOT / "reports" / f"run_logs_{tier}"

    if not run_dir.exists():
        st.warning(f"Not found: {run_dir}")
    else:
        files = sorted(
            [p for p in run_dir.glob("*") if p.suffix in {".md", ".json"}],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            st.info("No .md / .json artifacts found yet. Run an eval first.")
        else:
            file_pick = st.selectbox(
                "Select file",
                [p.name for p in files],
                index=0,
                key="artifact_file",
            )
            picked_path = run_dir / file_pick

            cA, cB = st.columns([0.7, 0.3])
            with cA:
                st.markdown(f"**Selected:** `{picked_path}`")
            with cB:
                try:
                    st.download_button(
                        "Download",
                        data=picked_path.read_bytes(),
                        file_name=picked_path.name,
                        use_container_width=True,
                    )
                except Exception:
                    pass

            st.write("")
            if picked_path.suffix == ".json":
                try:
                    st.json(json.loads(picked_path.read_text(encoding="utf-8")))
                except Exception:
                    st.code(picked_path.read_text(encoding="utf-8", errors="ignore"))
            else:
                # .md
                text = picked_path.read_text(encoding="utf-8", errors="ignore")
                # Render markdown, and also keep a raw view expander
                st.markdown(text)
                with st.expander("Raw text"):
                    st.code(text)

    st.markdown("</div>", unsafe_allow_html=True)
