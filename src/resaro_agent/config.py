"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # Prefer RESARO_* env vars, fall back to older names for compatibility.
    llm_provider: str = os.getenv("RESARO_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "mock")).strip().lower()  # hf|mock

    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    hf_model_id: str = os.getenv("RESARO_HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    data_dir: str = os.getenv("DATA_DIR", "src/data")
    template_path: str = os.getenv("TEMPLATE_PATH", "src/data/templates/briefing_template.md")

    # V0 budgets (keep tight for cost/latency)
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "600"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    retry_budget: int = int(os.getenv("RETRY_BUDGET", "1"))  # V0: minimal retry for validation failures
    
SETTINGS = Settings()
