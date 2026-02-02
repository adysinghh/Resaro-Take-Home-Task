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
```

I started V0 with the noraml implementation following the foundational Planning and reasoning Method - ReAct: which was made after the realisation that the system at that time followed the bad way of reasoning that is it was fragmanted into two ways which was 'Thinker' and 'Doer'; Thinker was usually following the COT approach and Doer just completed the given task but then they never worked in sync; so ReAct  both approch which was: Thought -> Action -> then the new Thought was shaped by the output of the last action; which followed the [interleaved reasoning and acting]
