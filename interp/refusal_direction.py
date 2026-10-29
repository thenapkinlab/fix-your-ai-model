#!/usr/bin/env python3
"""Refusal direction (Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction") for Qwen 2.5 1.5B,
then: how far does each XSTest prompt point along it, before and after our LoRA fine-tunes?

Direction = mean last-token activation (harmful) - mean (harmless), per layer, from the BASE model.
Prompts for the direction come from the training pools, never from XSTest:
  harmful  : finetune/v2/pools/harm_short_refusals.jsonl (BeaverTails / Do-Not-Answer, refused by base)
  harmless : Tulu 3 human-written chat (oasst1 / no_robots)
Layer = best harmful/harmless separation on a held-out half of those prompts.

Writes interp/results/acts_<model>.npz (XSTest last-token activations, every layer) and direction.npz.
Usage: python -I interp/refusal_direction.py
"""
import glob, json, os, random
import numpy as np, pandas as pd
import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 256                                            # prompts per side (half to build, half to pick the layer)
MODELS = {"base": None,
          "v1_n250": "finetune/adapters/seed2/n250",
          "v1_n1500": "finetune/adapters/seed2/n1500",
          "v2_n1500": "finetune/v2/adapters/seed2/n1500"}

def last_token_acts(model, tok, prompts):
    """Residual stream at the final prompt token after every layer: (n_prompts, n_layers, d)."""
    out = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        h = model.model.embed_tokens(mx.array([ids]))
        mask = create_attention_mask(h, None)
        per_layer = []
        for layer in model.model.layers:
            h = layer(h, mask, None)
            per_layer.append(h[0, -1])
        out.append(np.array(mx.stack(per_layer).astype(mx.float32)))
    return np.stack(out)

# ---------- prompts
rng = random.Random(0)
harm = [json.loads(l) for l in open(os.path.join(ROOT, "finetune/v2/pools/harm_short_refusals.jsonl"))]
harm = [r["prompt"] for r in harm if r["refused"]]
T = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--tulu-3-sft-mixture/snapshots/*/data/*.parquet")))
tl = pd.concat([pd.read_parquet(f) for f in T]); tl = tl[tl.source.str.contains("oasst1|no_robots")]
safe = [list(m)[0]["content"] for m in tl.messages if len(m) == 2 and 15 <= len(list(m)[0]["content"]) <= 300]
safe = [p for p in safe if p.isascii()]                    # English only, to match the harmful side
harm, safe = rng.sample(harm, N), rng.sample(safe, N)
xs = [dict(r) for r in load_dataset("walledai/XSTest", split="test")]

# ---------- direction from the base model
model, tok = load(MODEL)
H, S = last_token_acts(model, tok, harm), last_token_acts(model, tok, safe)
half = N // 2
d = H[:half].mean(0) - S[:half].mean(0)                    # (layers, dim)
u = d / np.linalg.norm(d, axis=-1, keepdims=True)
def sep(L):                                                # Cohen's d of projections on the held-out half
    a, b = H[half:, L] @ u[L], S[half:, L] @ u[L]
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
seps = [sep(L) for L in range(u.shape[0])]
best = int(np.argmax(seps))
print("separation by layer:", " ".join(f"{L}:{s:.1f}" for L, s in enumerate(seps)))
print("chosen layer:", best)
np.savez(os.path.join(OUT, "direction.npz"), u=u, seps=np.array(seps), best=best)
del model

# ---------- XSTest activations for every model
for name, ad in MODELS.items():
    path = os.path.join(OUT, f"acts_{name}.npz")
    if os.path.exists(path): continue
    if ad and not os.path.exists(os.path.join(ROOT, ad)):     # fine-tuned adapters are optional: train them first (finetune/)
        print("skipping", name, "(adapter not trained yet)"); continue
    model, tok = load(MODEL, adapter_path=os.path.join(ROOT, ad) if ad else None)
    A = last_token_acts(model, tok, [r["prompt"] for r in xs])
    np.savez_compressed(path, acts=A.astype(np.float16), prompt=[r["prompt"] for r in xs],
                        label=[r["label"] for r in xs], type=[r["type"] for r in xs])
    print("saved", name, A.shape, flush=True)
    del model
