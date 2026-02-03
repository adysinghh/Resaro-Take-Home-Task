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

Below are the results and Problems in this version that I encountered, along with the Proposed Solution (Our current setup follows a ReAct and COT setup, and I created a large corpus of corpus with 3 tiers to try to mimick real web noises):

| run_set | tier | n_runs | success_rate | leakage_rate | avg_tool_calls | avg_template_coverage | avg_products_f1 | avg_partnerships_f1 | injection_output_rate | suite_total_ms | avg_total_ms | p50_total_ms | p90_total_ms | avg_llm_tokens_est | avg_llm_ms | avg_llm_calls |
|---------|------|--------|--------------|--------------|----------------|----------------------|-----------------|---------------------|----------------------|----------------|--------------|--------------|--------------|-------------------|------------|---------------|
| 4-run | easy | 4 | 0.75 | 0.0 | 13.25 | 0.75 | 0.2222222222222222 | 0.3333333333333333 | 0.0 | 188681.0 | 47157.5 | 17404.0 | 22924.0 | 3365.25 | 14551.5 | 7.0 |
| 4-run | hard | 4 | 0.5 | 0.0 | 14.0 | 1.0 | 0.0 | 0.16666666666666666 | 0.0 | 57353.0 | 14324.5 | 15515.5 | 17117.0 | 3473.5 | 13013.25 | 7.25 |
| 4-run | realistic | 4 | 0.5 | 0.0 | 12.5 | 0.9583333333333334 | 0.5555555555555555 | 0.6666666666666666 | 0.0 | 59449.0 | 14844.75 | 14664.0 | 16625.0 | 3095.5 | 12709.75 | 6.5 |
| 58-run | easy | 58 | 0.8103448275862069 | 0.0 | 12.5 | 0.8793103448275862 | 0.4066666666666666 | 0.42 | 0.0 | | | | | | | |
| 58-run | hard | 58 | 0.603448275862069 | 0.0 | 12.844827586206897 | 0.7844827586206896 | 0.37333333333333335 | 0.3806666666666667 | 0.0 | | | | | | | |
| 58-run | realistic | 58 | 0.7586206896551724 | 0.0 | 12.379310344827585 | 0.8505747126436782 | 0.4666666666666666 | 0.4893333333333333 | 0.0 | | | | | | | |

## Problems that I Encountered

> **Note:** Formatting + emphasis only. Text unchanged.

---

### 1) **No Persistant Learning (Major)**

Agent correct issues via heuristics, but doesn't lear pattern from past error and then apply in the run. <br>

**ROOT CAUSE:** We Dont have any persistant memory!;
**Soln:** 1.1 Use an optimized memory such as **'SimpleMemory (2025)'** this is an improvement on the **'Mem0'** which is an YC startup; In this we could store the sucess cases, what worked and what did not.

---

### 2) **Single Small LLM does: Planning + deciding + transaltion. (Quality, Speed botteneck) (Major)** <br>

**Evidence:** Latency is high, I ran all these tests on the Macbook Pro with M3 pro chip.

We could fix this by using **2 LLMs**, to split the task between 2 llms like done in modern agentic systems; where we could use an **'Small Model'** for **Control (Plan)** and then use **'Large Model'** for the other task such as; **translate, Rewrite and Summarize**

Another Fix is to use **'Prompt Comperession'**, where we could summarizr state instead of feeding large JSON or DOC and feeding the entire data into the LLM.

---

### 3) **Third biggest problem in current setup is 'EFFICIENCY'(Major)**

#### 3.1. **ReAct drawback + loops**

ReAct exhibits a major drawback, that it Plans -> Action -> Use output to shape next plan; but it doesnt caters the **Reflection** i.e., current state of the system, how close it is to the goal; which sometimes leads the system in the loop which is beutifully explained in the **ReflAct Paper (2025, ACL)**.

**Solution** to this is to migrate the Reasoning phase from **ReAct -> ReflAct**, which will stop the looping since it knows, where it is, whats done and what remains.

After each step, Run **tiny-reflection loop** like -

* "Did we progress stage?"
* "Is next tool valid under current stage"
* "Did we hit repetation".

#### 3.2. **High 'avg_tool_calls'**

To fix this we could follow recent Methods which adds an **rewarding function** in between to maximise the output, here we could learn **Policies using reward signal** which are reported in the JSON for each run **[Sucess, Latency, tool_call/ F1]**

---

### 4) **Agents Loops/ Redundant tool calls**

* 4.1.
ReAct relies mostly on prompt, but doesnt hard block any loops now.

* 4.2.
When LLM output is invalid json/ missing req. fields: Fallback policies and selct tools in the way that creates repetible loops.

* 4.3.
Controller doesnt have a **'Notion of stage completeness'**, i.e., Profile done, web search done (which is also a Reflection signal).

Most of this could be fixed by adding **ToolRM** and **ReflAct**, a **self correctness**.

---

### 5) **Successs rate failure**

We could use **Determinstic doc assembly** (Reduce relicance on the LLM for structuring)

**Stronger Stop Condition:** Dont allow stopping till it (LLM) fixes the structure of final doc.

---

### 6) **Tool Args/ Schema failure**

LLM sometimes emits incomplete tool args, but current systems doesnt doesnt reliably **"Self-Repair"** tools args.

So now we have fallback paths, we need to replace that by **self-correction loops**.

**Solution:**

* 6.1.2.
If tools args missing, then produce an **"error observation"** & force a correction loops.

* 6.1.2.
for generate_doc, auto injects a **default template** if missing.

* 6.2.1. **(Self correction loop)**
Add **'Tool Error Reflection'**: Put structured tool_error object into state **(tool_name, missing_key)** & feed it into next LLM output; which requires next LLM output to fix args before any other action
