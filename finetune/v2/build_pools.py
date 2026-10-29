#!/usr/bin/env python3
"""v2 data, step 1: collect SHORT harmful questions, the shape v1's refusal data was missing.

v1 problem: 66% of benign->answer prompts were short questions, but only 2% of harmful->refuse prompts were
(WildJailbreak's harmful prompts are long "write/create..." requests). The model learned "short question -> answer".

Sources (free):
  BeaverTails 330k (PKU)       : prompts with >= 2 responses, ALL labelled unsafe (prompt is clearly harmful)
  Do-Not-Answer (LibrAI)       : malicious uses, discrimination/toxicity, private-individual privacy
Same XSTest contamination filter as v1. Output: v2/pools/harm_short_candidates.jsonl (prompts only;
refusals come from the base model itself in gen_refusals.py, so we don't teach a new refusal style).
"""
import collections, json, os, random, re
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pools"); os.makedirs(OUT, exist_ok=True)

def words(t): return re.findall(r"[a-z']+", str(t).lower())
xs = [r["prompt"] for r in load_dataset("walledai/XSTest", split="test")]
xs_sets = [set(words(p)) for p in xs]
xs_grams = {" ".join(w[i:i + 5]) for w in map(words, xs) for i in range(len(w) - 4)}
def near_xstest(p):
    w = words(p); s = set(w)
    if any(" ".join(w[i:i + 5]) in xs_grams for i in range(len(w) - 4)): return True
    return any(len(s & x) / max(1, len(s | x)) >= 0.5 for x in xs_sets)      # stricter than v1 (0.6): short prompts

def short_q(p): return len(p.split()) <= 12 and p.strip().endswith("?")

stats, cands = {}, {}
bt = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
safe = collections.defaultdict(list); cat = {}
for r in bt:
    safe[r["prompt"]].append(r["is_safe"])
    cat.setdefault(r["prompt"], [k for k, v in r["category"].items() if v])
bt_harm = [p for p, v in safe.items() if len(v) >= 2 and not any(v) and 15 <= len(p) <= 300]
stats["beavertails_all_unsafe"] = len(bt_harm)
for p in bt_harm: cands[p.strip()] = dict(source="beavertails", category=",".join(cat[p]))

dna = load_dataset("LibrAI/do-not-answer", split="train")
KEEP = ("Malicious Uses", "Discrimination, Exclusion, Toxicity, Hateful, Offensive")
for r in dna:
    if r["risk_area"] in KEEP or r["types_of_harm"].startswith("Compromise privacy"):
        cands[r["question"].strip()] = dict(source="do-not-answer", category=r["types_of_harm"])
stats["do_not_answer_kept"] = sum(1 for v in cands.values() if v["source"] == "do-not-answer")

before = len(cands)
cands = {p: v for p, v in cands.items() if not near_xstest(p)}
stats["dropped_near_xstest"] = before - len(cands)
stats["kept"] = len(cands)
stats["kept_short_questions"] = sum(map(short_q, cands))

rows = [dict(prompt=p, short_q=short_q(p), **v) for p, v in cands.items()]
random.Random(0).shuffle(rows)
with open(os.path.join(OUT, "harm_short_candidates.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
json.dump(stats, open(os.path.join(OUT, "pool_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
