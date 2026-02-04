# scripts/pick_hf_model.py
import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
PROVIDER = os.getenv("RESARO_HF_PROVIDER", "hf-inference")

CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]


def try_model(mid: str) -> bool:
    client = InferenceClient(model=mid, provider=PROVIDER, token=HF_TOKEN)

    resp = client.chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly: OK_REAL_LLM"}],
        max_tokens=10,
        temperature=0.0,
    )

    out = resp.choices[0].message["content"].strip()
    print(mid, "->", out[:80])
    return out == "OK_REAL_LLM"




if __name__ == "__main__":
    assert HF_TOKEN, "HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) is missing."
    found = False

    for mid in CANDIDATES:
        try:
            if try_model(mid):
                print("\nWORKING_MODEL =", mid)
                found = True
                break
        except Exception as e:
            print(mid, "FAILED:", repr(e))
    
    if not found:
        raise SystemExit(1)
