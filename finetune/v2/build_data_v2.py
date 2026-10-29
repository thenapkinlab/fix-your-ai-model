#!/usr/bin/env python3
"""v2 data, step 3: assemble training sets where the question's SHAPE no longer predicts refuse vs answer.

Same three equal parts as v1 (harmful -> refusal, risky-sounding benign -> answer, general chat), plus:
  1. Shape matching: parts 1 and 2 get the same mix of short questions / long questions / requests.
     Short harmful questions come from v2 pools (BeaverTails + Do-Not-Answer, refused by the base model itself).
  2. Scary-word matching: inside short questions, both parts get the same share with a "scary" word
     (kill, shoot, steal...), and benign picks prefer the same words as the harmful ones (contrast pairs).
  3. Premise rejection counts as a good answer to a discriminatory question ("No, that is not true...").
Prints a shortcut check: how well shape alone predicts "refuse" in v1 vs v2.

Usage: python -I finetune/v2/build_data_v2.py   (after gen_refusals.py)
"""
import collections, glob, json, os, random, re
import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SIZES = [1500, 5000, 10000]
SEEDS = [0, 1, 2]
N_VALID = 60
MIX = {"short_q": 0.55, "long_q": 0.3, "request": 0.15}  # same for parts 1 and 2; benign requests are scarce (~500)
SCARY_SHARE = 0.2                                         # of short questions, in both parts (pool allows ~390 harmful)

REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i can not|i won'?t|i will not|i must (respectfully )?decline|"
                     r"i'?m (not able|unable)|i am (not able|unable)|as an ai|i apologi[sz]e|unfortunately,? i)", re.I)
FIRST = re.compile(r"\bI (cannot|can't|can not|am unable to|am not able to|won't|will not) (discuss|provide|assist|help|answer|comment|share|give|engage|fulfil)", re.I)
PREMISE = re.compile(r"^\W*(no[,.]|that('s| is) (not true|a (harmful )?stereotype|not accurate)|it('s| is) not (true|accurate|fair)|this is a (harmful )?stereotype)", re.I)
def refuses(a): return bool(REFUSAL.search(str(a)[:200]) or FIRST.search(str(a)[:250]))
SCARY = re.compile(r"\b(kill|killing|shoot|shooting|execute|attack|destroy|bomb|blow up|murder|strangle|crush|eliminate|slaughter|butcher|"
                   r"choke|stab|poison|drown|hunt|eradicate|wipe out|decimate|blast|assassinate|terminate|nuke|smash|steal|hijack|torch|"
                   r"burn|capture|liquidate|throttle|coke|crack|hack|rob|insult)\b", re.I)

def shape(p):
    p = str(p).strip()
    if not p.endswith("?"): return "request"
    return "short_q" if len(p.split()) <= 12 else "long_q"
def scary(p):
    m = SCARY.search(str(p)); return m.group(1).lower() if m else None

def words(t): return re.findall(r"[a-z']+", str(t).lower())
xs = [r["prompt"] for r in load_dataset("walledai/XSTest", split="test")]
xs_sets = [set(words(p)) for p in xs]
xs_grams = {" ".join(w[i:i + 5]) for w in map(words, xs) for i in range(len(w) - 4)}
def near_xstest(p):
    w = words(p); s = set(w)
    if any(" ".join(w[i:i + 5]) in xs_grams for i in range(len(w) - 4)): return True
    return any(len(s & x) / max(1, len(s | x)) >= 0.5 for x in xs_sets)

# ---------- pools
W = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--wildjailbreak/snapshots/*/train/train.tsv"))[0]
wj = pd.read_csv(W, sep="\t", on_bad_lines="skip", usecols=["vanilla", "completion", "data_type"]).dropna().drop_duplicates("vanilla")
wj = wj[(wj.vanilla.str.len().between(15, 300)) & (wj.completion.str.len().between(40, 1200))]
def wj_pool(dtype, want_refusal):
    d = wj[(wj.data_type == dtype) & (wj.completion.map(refuses) == want_refusal)]
    return [dict(prompt=r.vanilla, answer=r.completion, src="wildjailbreak") for r in d.itertuples() if not near_xstest(r.vanilla)]
harmful = wj_pool("vanilla_harmful", True)
benign = wj_pool("vanilla_benign", False)
gen = [json.loads(l) for l in open(os.path.join(HERE, "pools", "harm_short_refusals.jsonl"))]
harmful += [dict(prompt=r["prompt"], answer=r["answer"], src=r["source"]) for r in gen
            if refuses(r["answer"]) or PREMISE.search(r["answer"])]

T = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--tulu-3-sft-mixture/snapshots/*/data/*.parquet")))
tl = pd.concat([pd.read_parquet(f) for f in T]); tl = tl[tl.source.str.contains("oasst1|no_robots")]
general = []
for m in tl.messages:
    m = list(m)
    if len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
        q, a = m[0]["content"], m[1]["content"]
        if 15 <= len(q) <= 300 and 40 <= len(a) <= 1200 and not refuses(a) and not near_xstest(q):
            general.append(dict(prompt=q, answer=a, src="general"))

def strata(pool):  # (shape, scary?) -> list
    s = collections.defaultdict(list)
    for x in pool: s[(shape(x["prompt"]), scary(x["prompt"]) is not None)].append(x)
    return s

rng = random.Random(1234)
for p in (harmful, benign, general): rng.shuffle(p)
valid = harmful[:N_VALID] + benign[:N_VALID] + general[:N_VALID]
H, B, G = strata(harmful[N_VALID:]), strata(benign[N_VALID:]), general[N_VALID:]
print("available harmful:", {k: len(v) for k, v in sorted(H.items())})
print("available benign: ", {k: len(v) for k, v in sorted(B.items())})

def quota(k):  # items per (shape, scary) stratum for a part of size k
    q = {}
    for s, f in MIX.items():
        n = round(k * f)
        if s == "short_q":
            q[(s, True)] = round(n * SCARY_SHARE); q[(s, False)] = n - q[(s, True)]
        else:
            q[(s, False)] = n                                 # scary words outside short questions: not controlled
    return q

def order_benign_scary(benign_list, harm_list):
    """Contrast pairs: benign scary short questions that share a word with a harmful one go first, in step."""
    by_word = collections.defaultdict(list)
    for x in benign_list: by_word[scary(x["prompt"])].append(x)
    out, used = [], set()
    for h in harm_list:
        lst = by_word.get(scary(h["prompt"]), [])
        while lst and id(lst[0]) in used: lst.pop(0)
        if lst: out.append(lst[0]); used.add(id(lst.pop(0)))
    return out + [x for x in benign_list if id(x) not in used]

def msg(x): return {"messages": [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["answer"]}]}
def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(msg(r)) + "\n")

def shortcut_report(rows, label):
    """How often is the answer a refusal, given only the prompt's shape? Equal across shapes = no shortcut."""
    t = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        key = shape(r["prompt"]) + ("+scary" if scary(r["prompt"]) else "")
        t[key][0] += refuses(r["answer"]); t[key][1] += 1
    print(f"  {label:<22}" + "  ".join(f"{k}: {100*a/b:4.0f}% refuse (n={b})" for k, (a, b) in sorted(t.items())))

# v1 reference
v1 = [json.loads(l)["messages"] for l in open(os.path.join(HERE, "..", "data", "seed2", "n1500", "train.jsonl"))]
print("\nshortcut check (share of examples that are refusals, by prompt shape)")
shortcut_report([dict(prompt=m[0]["content"], answer=m[1]["content"]) for m in v1], "v1 n1500")

for seed in SEEDS:
    r = random.Random(seed)
    hs = {k: r.sample(v, len(v)) for k, v in H.items()}
    bs = {k: r.sample(v, len(v)) for k, v in B.items()}
    bs[("short_q", True)] = order_benign_scary(bs[("short_q", True)], hs[("short_q", True)])
    gs = r.sample(G, len(G))
    for n in SIZES:
        k = n // 3
        rows = []
        for key, q in quota(k).items():
            if len(hs.get(key, [])) < q or len(bs.get(key, [])) < q:
                raise SystemExit(f"not enough data for {key} at n={n}: harmful {len(hs.get(key, []))}, benign {len(bs.get(key, []))}, need {q}")
            rows += hs[key][:q] + bs[key][:q]                 # prefixes -> nested across sizes
        rows += gs[:n - len(rows)]
        random.Random(seed * 100 + n).shuffle(rows)
        d = os.path.join(OUT, f"seed{seed}", f"n{n}")
        write(os.path.join(d, "train.jsonl"), rows); write(os.path.join(d, "valid.jsonl"), valid)
        if seed == 0: shortcut_report(rows, f"v2 n{n}")
print("wrote", OUT)
