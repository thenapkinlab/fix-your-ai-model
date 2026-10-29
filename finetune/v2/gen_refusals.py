#!/usr/bin/env python3
"""v2 data, step 2: get the BASE model's own refusal for each short harmful question.

Why the base model: it already refuses ~99% of harmful prompts, so its refusals keep the model's natural style.
Borrowing another model's refusals would add a new "refusal voice" the model could latch onto as a shortcut.
Prompts the base model does NOT refuse are dropped (a second check that the prompt really is harmful).

Usage (resumable; ~3 s per prompt on a 16 GB MacBook):
  python -I finetune/v2/gen_refusals.py --limit 5      # smoke test
  caffeinate -i python -I finetune/v2/gen_refusals.py  # all candidates (~3 h)
"""
import argparse, json, os, re
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pools", "harm_short_candidates.jsonl")
OUT = os.path.join(HERE, "pools", "harm_short_refusals.jsonl")

REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i can not|i won'?t|i will not|i must (respectfully )?decline|"
                     r"i'?m (not able|unable)|i am (not able|unable)|as an ai|i apologi[sz]e|unfortunately,? i)", re.I)
FIRST = re.compile(r"\bI (cannot|can't|can not|am unable to|am not able to|won't|will not) (discuss|provide|assist|help|answer|comment|share|give|engage|fulfil)", re.I)
def refuses(a): return bool(REFUSAL.search(a[:200]) or FIRST.search(a[:250]))

def trim(a):  # cut a max_tokens-truncated answer back to its last full sentence
    a = a.strip()
    if a.endswith((".", "!", "?")): return a
    cut = max(a.rfind(". "), a.rfind("! "), a.rfind("? "))
    return a[:cut + 1] if cut > 0 else a

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

rows = [json.loads(l) for l in open(SRC)][:a.limit]
done = {json.loads(l)["prompt"] for l in open(OUT)} if os.path.exists(OUT) else set()
model, tok = load(a.model)
kept = 0
with open(OUT, "a") as f:
    for i, r in enumerate(rows):
        if r["prompt"] in done: continue
        prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        ans = trim(generate(model, tok, prompt=prompt, max_tokens=160, verbose=False))
        ok = refuses(ans); kept += ok
        f.write(json.dumps(dict(r, answer=ans, refused=ok)) + "\n"); f.flush()
        if i % 50 == 0: print(f"{i}/{len(rows)}  refused so far this run: {kept}", flush=True)
print("done:", OUT)
