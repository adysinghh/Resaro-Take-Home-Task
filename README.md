# Resaro Take-home — Agentic LLM + Testing 
This will contain the Research work that I went through and used for Planning and Building the final version!

## (V0)
## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/generate_synth_data.py
python scripts/run_eval.py