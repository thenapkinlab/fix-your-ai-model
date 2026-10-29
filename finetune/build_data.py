#!/usr/bin/env python3
"""Build nested LoRA training sets for the safety sweep.

Pools (all free, non-profit sources):
  harmful -> refusal   : WildJailbreak 'vanilla_harmful' rows whose completion is a refusal
  benign  -> helpful   : WildJailbreak 'vanilla_benign' rows whose completion is NOT a refusal
  general              : Tulu 3 SFT mix, human-written subsets (oasst1, no_robots), single-turn

Steps: clean -> drop anything close to XSTest -> carve fixed validation + held-out eval -> nested,
stratified subsets (250/500/1000/1500, equal thirds) for 3 seeds. Output is mlx-lm chat format.
"""
import glob, json, os, random, re
import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SIZES = [250, 500, 1000, 1500]
SEEDS = [0, 1, 2]
N_VALID, N_EVAL = 60, 100           # per category (valid = loss tracking; eval = sweep metric, prompts only)

REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i can not|i won'?t|i will not|i must (respectfully )?decline|"
                     r"i'?m (not able|unable)|i am (not able|unable)|as an ai|i apologi[sz]e|unfortunately,? i)", re.I)
def refuses(text): return bool(REFUSAL.search(str(text)[:200]))

def words(t): return re.findall(r"[a-z']+", str(t).lower())

# ---------- XSTest contamination filter
xs = [r["prompt"] for r in load_dataset("walledai/XSTest", split="test")]
xs_sets = [set(words(p)) for p in xs]
xs_grams = set()
for p in xs:
    w = words(p)
    xs_grams.update(" ".join(w[i:i + 5]) for i in range(len(w) - 4))
def near_xstest(prompt):
    w = words(prompt); s = set(w)
    if any(" ".join(w[i:i + 5]) in xs_grams for i in range(len(w) - 4)): return True
    return any(len(s & x) / max(1, len(s | x)) >= 0.6 for x in xs_sets)

# ---------- load pools
W = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--wildjailbreak/snapshots/*/train/train.tsv"))[0]
wj = pd.read_csv(W, sep="\t", on_bad_lines="skip", usecols=["vanilla", "completion", "data_type"])
wj = wj.dropna().drop_duplicates("vanilla")
wj = wj[(wj.vanilla.str.len().between(15, 300)) & (wj.completion.str.len().between(40, 1200))]

stats = {}
def pool(dtype, want_refusal):
    d = wj[wj.data_type == dtype]
    d = d[d.completion.map(refuses) == want_refusal]
    before = len(d)
    d = d[~d.vanilla.map(near_xstest)]
    stats[dtype] = dict(after_refusal_filter=before, dropped_near_xstest=before - len(d), kept=len(d))
    return [dict(prompt=r.vanilla, answer=r.completion) for r in d.itertuples()]

harmful = pool("vanilla_harmful", True)
benign = pool("vanilla_benign", False)

T = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--tulu-3-sft-mixture/snapshots/*/data/*.parquet")))
tl = pd.concat([pd.read_parquet(f) for f in T])
tl = tl[tl.source.str.contains("oasst1|no_robots")]
general = []
for m in tl.messages:
    m = list(m)
    if len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
        q, a = m[0]["content"], m[1]["content"]
        if 15 <= len(q) <= 300 and 40 <= len(a) <= 1200 and not refuses(a) and not near_xstest(q):
            general.append(dict(prompt=q, answer=a))
stats["general"] = dict(kept=len(general))

# ---------- fixed validation + held-out eval (same for every seed and size)
rng = random.Random(1234)
for p in (harmful, benign, general): rng.shuffle(p)
valid = harmful[:N_VALID] + benign[:N_VALID] + general[:N_VALID]
heldout = [dict(prompt=x["prompt"], label="harmful") for x in harmful[N_VALID:N_VALID + N_EVAL]] + \
          [dict(prompt=x["prompt"], label="benign") for x in benign[N_VALID:N_VALID + N_EVAL]]
cut = N_VALID + N_EVAL
H, B, G = harmful[cut:], benign[cut:], general[N_VALID:]

def msg(x): return {"messages": [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["answer"]}]}
def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(msg(r)) + "\n")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "heldout_eval.jsonl"), "w") as f:
    for r in heldout: f.write(json.dumps(r) + "\n")

# ---------- nested, stratified subsets per seed
for seed in SEEDS:
    r = random.Random(seed)
    h, b, g = H[:], B[:], G[:]
    r.shuffle(h); r.shuffle(b); r.shuffle(g)
    for n in SIZES:
        k = n // 3
        rows = h[:k] + b[:k] + g[:n - 2 * k]          # nested: smaller sizes are prefixes of larger ones
        r2 = random.Random(seed * 100 + n); r2.shuffle(rows)
        d = os.path.join(OUT, f"seed{seed}", f"n{n}")
        write(os.path.join(d, "train.jsonl"), rows)
        write(os.path.join(d, "valid.jsonl"), valid)

json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
print("wrote", OUT)
