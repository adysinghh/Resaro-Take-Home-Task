from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from .config import SETTINGS

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os, time, json, re
from dataclasses import dataclass
from typing import Optional, List, Dict

STRICT_LLM = os.getenv("RESARO_STRICT_LLM", "0") == "1"

# LLM backend controls:
#   local  -> always local transformers
#   hf     -> always remote HF router/provider
#   auto   -> try remote; if 402/payment -> local; else error (if strict)
LLM_BACKEND = os.getenv("RESARO_LLM_BACKEND", "auto").strip().lower()

HF_PROVIDER = os.getenv("RESARO_HF_PROVIDER", "hf-inference").strip()
HF_MODEL_ID = os.getenv("RESARO_HF_MODEL_ID", "") or SETTINGS.hf_model_id
HF_TOKEN = SETTINGS.hf_token

STRICT_LLM = os.getenv("RESARO_STRICT_LLM", "0") == "1"
LLM_BACKEND = os.getenv("RESARO_LLM_BACKEND", "").strip().lower()

LOCAL_MODEL_ID = os.getenv("RESARO_LOCAL_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct").strip()
LOCAL_DEVICE = os.getenv("RESARO_LOCAL_DEVICE", "auto").strip().lower()  # auto|mps|cpu|cuda
LOCAL_TRUNCATE_CHARS = int(os.getenv("RESARO_LOCAL_TRUNCATE_CHARS", "20000"))  # safety


@dataclass
class LLMResult:
    text: str
    tokens_estimate: int
    latency_ms: int

def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)

def _truncate_prompt(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    # keep the end (often contains the actual instruction)
    return s[-limit:]



class BaseLLM:
    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> LLMResult:
        raise NotImplementedError


class MockLLM(BaseLLM):
    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> LLMResult:
        t0 = time.time()

        if "Return a JSON plan" in prompt:
            text = json.dumps(
                {"steps": [
                    "get_company_info(company_name)",
                    "mock_web_search(company_name)",
                    "generate_document(template, facts)",
                    "translate_document(document, target_language) if needed",
                    "security_filter(document)"
                ]},
                indent=2,
            )
        elif "Translate the document" in prompt:
            m = re.search(r"TARGET_LANGUAGE:\s*(.+)", prompt)
            lang = (m.group(1).strip() if m else "Unknown")
            prefix = {
                "german": "Dies ist eine Übersetzung. ",
                "french": "Ceci est une traduction. ",
                "spanish": "Esta es una traducción. ",
            }.get(lang.lower(), f"[TRANSLATED TO {lang}] ")
            body = prompt.split("DOCUMENT_START", 1)[-1].split("DOCUMENT_END", 1)[0].strip()
            text = prefix + body
        else:
            text = "DRAFT:\n" + prompt[-min(len(prompt), 800):]

        latency_ms = int((time.time() - t0) * 1000)
        return LLMResult(text=text, tokens_estimate=_estimate_tokens(prompt) + _estimate_tokens(text), latency_ms=latency_ms)

# add near top
from .trace import trace_llm

class TracedLLM(BaseLLM):
    def __init__(self, inner: BaseLLM, name: str):
        self.inner = inner
        self.name = name

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> LLMResult:
        res = self.inner.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        trace_llm(
            "llm_call",
            prompt=prompt,
            completion=res.text,
            model=self.name,
            latency_ms=res.latency_ms,
            tokens_est=res.tokens_estimate,
        )
        return res


@lru_cache(maxsize=2)
def _load_local_model(model_id: str):

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # device selection
    if LOCAL_DEVICE != "auto":
        device = LOCAL_DEVICE
    else:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # dtype selection
    if device in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tok, model, device


class LocalTransformersLLM:
    _tokenizer = None
    _model = None
    _device = None

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._load_once()

    def _load_once(self):
        if LocalTransformersLLM._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # device selection
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        LocalTransformersLLM._device = device

        tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        # --- attention_mask/pad fix (also solves your warning) ---
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()

        # make sure model knows pad token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id

        # move model to device
        if device != "cpu":
            model.to(device)

        LocalTransformersLLM._tokenizer = tok
        LocalTransformersLLM._model = model

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> LLMResult:
        import torch

        t0 = time.time()
        prompt = _truncate_prompt(prompt, LOCAL_TRUNCATE_CHARS)

        tok = LocalTransformersLLM._tokenizer
        model = LocalTransformersLLM._model
        device = LocalTransformersLLM._device

        # --- chat_template ---
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow instructions exactly."},
            {"role": "user", "content": prompt},
        ]
        chat_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        enc = tok(chat_str, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        if device != "cpu":
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=float(temperature),
            pad_token_id=tok.pad_token_id,
        )

        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # decode only newly generated part
        new_tokens = out_ids[0, input_ids.shape[-1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()

        latency_ms = int((time.time() - t0) * 1000)
        return LLMResult(text=text, tokens_estimate=_estimate_tokens(prompt) + _estimate_tokens(text), latency_ms=latency_ms)

class HuggingFaceInferenceLLM(BaseLLM):
    """
    Remote HF router/provider via huggingface_hub InferenceClient.
    """
    def __init__(self, token: str, model_id: str):
        from huggingface_hub import InferenceClient
        self.token = token
        self.model_id = model_id
        self.client = InferenceClient(
            model=self.model_id,
            provider=HF_PROVIDER,
            token=self.token,
        )

    def _is_payment_error(self, e: Exception) -> bool:
        msg = str(e).lower()
        return ("402" in msg) or ("payment required" in msg) or ("credit balance is depleted" in msg)

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> LLMResult:
        import torch

        t0 = time.time()

        # avoid accidental huge prompts blowing up CPU/MPS memory
        if len(prompt) > LOCAL_TRUNCATE_CHARS:
            prompt = prompt[-LOCAL_TRUNCATE_CHARS:]

        # If tokenizer supports chat templates, use them (better for instruct models)
        if getattr(self.tok, "chat_template", None) and hasattr(self.tok, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            rendered = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = self.tok(rendered, return_tensors="pt")
        else:
            enc = self.tok(prompt, return_tensors="pt")

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        input_len = input_ids.shape[-1]

        do_sample = temperature is not None and temperature > 1e-6

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,  # important for stable generation
        )

        if do_sample:
            gen_kwargs.update(dict(
                temperature=float(temperature),
                top_p=0.9,
            ))
        else:
            # Avoid warnings if model's generation_config has sampling defaults
            gen_kwargs.update(dict(
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            ))

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

        gen_ids = out_ids[0, input_len:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        latency_ms = int((time.time() - t0) * 1000)
        tokens_est = int(input_len + gen_ids.numel())
        return LLMResult(text=text, tokens_estimate=tokens_est, latency_ms=latency_ms)



from .trace import get_trace  # optional, not required

@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    # decide backend exactly as you do now...
    if LLM_BACKEND == "local":
        base = LocalTransformersLLM(LOCAL_MODEL_ID)
        return TracedLLM(base, name=f"local:{LOCAL_MODEL_ID}")

    if LLM_BACKEND == "hf":
        base = HuggingFaceInferenceLLM(token=HF_TOKEN, model_id=HF_MODEL_ID)
        return TracedLLM(base, name=f"hf:{HF_PROVIDER}:{HF_MODEL_ID}")

    # auto...
    if SETTINGS.llm_provider == "hf" and HF_TOKEN:
        base = HuggingFaceInferenceLLM(token=HF_TOKEN, model_id=HF_MODEL_ID)
        return TracedLLM(base, name=f"hf:{HF_PROVIDER}:{HF_MODEL_ID}")

    base = LocalTransformersLLM(LOCAL_MODEL_ID)
    return TracedLLM(base, name=f"local:{LOCAL_MODEL_ID}")



import json

def safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()

    decoder = json.JSONDecoder()

    # Find the first '{' and attempt raw_decode from there.
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(s[i:])
                return obj if isinstance(obj, dict) else None
            except Exception:
                continue

    return None
