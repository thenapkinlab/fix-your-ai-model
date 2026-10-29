#!/usr/bin/env python3
"""Watch the refusal-direction position move during training, checkpoint by checkpoint (saved every 100 steps).
Forward passes only (no generation): a cheap "safety dial" to read while a fine-tune runs.

Usage: python -I interp/checkpoints.py      (after refusal_direction.py)
Writes interp/results/checkpoints.json
"""
import glob, json, os, shutil, sys, tempfile
import numpy as np
from datasets import load_dataset
from mlx_lm import load

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                   # python -I leaves the script folder off sys.path
from refusal_direction_lib import last_token_acts
L = 20
RUNS = {"v1_n1500": "finetune/adapters/seed2/n1500", "v2_n1500": "finetune/v2/adapters/seed2/n1500"}

u = np.load(os.path.join(HERE, "results", "direction.npz"))["u"][L]
xs = [dict(r) for r in load_dataset("walledai/XSTest", split="test")]
lab = np.array([r["label"] for r in xs])
base = np.load(os.path.join(HERE, "results", "acts_base.npz"))["acts"][:, L].astype(np.float32) @ u
lo, hi = base[lab == "safe"].mean(), base[lab == "unsafe"].mean()

out = {}
for name, ad in RUNS.items():
    ad = os.path.join(ROOT, ad)
    if not os.path.isdir(ad): print("skipping", name, "(adapter not trained yet)"); continue
    ckpts = sorted(glob.glob(os.path.join(ad, "0*_adapters.safetensors"))) + [os.path.join(ad, "adapters.safetensors")]
    out[name] = [dict(step=0, harmful=1.0, harmless=0.0)]
    for c in ckpts:
        step = int(os.path.basename(c)[:7]) if os.path.basename(c)[0] == "0" else 750
        tmp = tempfile.mkdtemp()
        shutil.copy(os.path.join(ad, "adapter_config.json"), tmp)
        os.symlink(c, os.path.join(tmp, "adapters.safetensors"))
        model, tok = load("Qwen/Qwen2.5-1.5B-Instruct", adapter_path=tmp)
        p = (last_token_acts(model, tok, [r["prompt"] for r in xs], layers=[L])[:, 0] @ u - lo) / (hi - lo)
        out[name].append(dict(step=step, harmful=float(p[lab == "unsafe"].mean()), harmless=float(p[lab == "safe"].mean())))
        print(name, out[name][-1], flush=True)
        shutil.rmtree(tmp); del model
json.dump(out, open(os.path.join(HERE, "results", "checkpoints.json"), "w"), indent=1)
