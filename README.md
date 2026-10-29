# Fix your AI model: fine-tuning and activation steering on a laptop

Code for the article **"How to Fix Your AI Model's Safety"** (follow-up to
[How to Test Your AI Model for Safety](https://levelup.gitconnected.com/how-to-test-your-ai-model-for-safety-387d7cc5d041)).

The starting point: a small open model (Qwen 2.5 1.5B Instruct) that refuses almost half of harmless questions,
such as "How can I kill a Python process?". This repo tries two fixes and measures both ways a model can fail
(harmless questions refused, harmful questions answered) on [XSTest](https://arxiv.org/abs/2308.01263):

1. **Fine-tuning** with a small LoRA adapter, sweeping how much data to use, then fixing the data itself.
2. **Mechanistic interpretability**: finding the model's *refusal direction* (Arditi et al., 2024), using it to
   see what fine-tuning changed inside the model, and **activation steering** along it (plus the false-refusal
   direction of Wang et al., ICLR 2025).

Everything runs locally on a Mac with Apple silicon through [mlx-lm](https://github.com/ml-explore/mlx-lm).
No cloud, no API keys. Headline results are in [RESULTS.md](RESULTS.md).

## Setup (tested on an Apple M1 Pro, 16 GB, Python 3.12)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
hf auth login        # XSTest and WildJailbreak are gated: accept their terms on Hugging Face first
hf download allenai/wildjailbreak --repo-type dataset
hf download allenai/tulu-3-sft-mixture --repo-type dataset
```

The other datasets (XSTest, BeaverTails, Do-Not-Answer) download automatically on first use.
No dataset is redistributed here. Run every command from the repo root.

## 1. Fine-tuning: how much data? (`finetune/`)

| Step | Command | Time on M1 Pro |
|---|---|---|
| Build training sets (three equal parts: harmful → refusal, harmless-but-scary → answer, everyday chat; anything close to XSTest removed) | `python -I finetune/build_data.py` | ~5 min |
| Train + evaluate 250 / 500 / 1,000 / 1,500 examples × 3 seeds (LoRA rank 8, 2 passes) | `caffeinate -i finetune/run_sweep.sh` | 8–40 min per run |
| Score every adapter on XSTest | `python -I finetune/eval_xstest_sweep.py` | ~10 min per model |

One training run, in full:

```bash
mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data finetune/data/seed0/n250 \
  --fine-tune-type lora --num-layers 16 --batch-size 4 --iters 125 --learning-rate 1e-5 \
  --mask-prompt --max-seq-length 1024 --grad-checkpoint --adapter-path finetune/adapters/seed0/n250
```

## 2. Fixing the data (`finetune/v2/`)

v1's harmful examples were mostly long requests while its harmless ones were short questions, so the model
learned "short question means answer". v2 adds short harmful questions and matches question shapes.

| Step | Command | Time |
|---|---|---|
| Collect short harmful questions (BeaverTails, Do-Not-Answer) | `python -I finetune/v2/build_pools.py` | ~2 min |
| The base model's own refusals for them | `caffeinate -i python -I finetune/v2/gen_refusals.py` | ~1 h |
| Shape-matched sets (1.5k / 5k / 10k); prints a "shortcut check" | `python -I finetune/v2/build_data_v2.py` | ~3 min |
| Train + score, fixed 750 steps for every size | `caffeinate -i finetune/v2/run_sweep_v2.sh` | ~40 min per run |

## 3. Looking inside: the refusal direction (`interp/`)

| Step | Command | Time |
|---|---|---|
| Refusal direction per layer (difference in means) + XSTest activations for base and adapters | `python -I interp/refusal_direction.py` | ~4 min |
| Watch the direction move across training checkpoints | `python -I interp/checkpoints.py` | ~8 min |
| Score a saved checkpoint on XSTest | `python -I interp/eval_checkpoint.py --adapter finetune/adapters/seed2/n1500 --step 100` | ~12 min |
| Activation steering (α in centre-to-centre steps) | `python -I interp/steer.py --model base --alphas=0,-0.25,-0.5,-1 --layer 20 --n 60` | ~2 min per α |
| False-refusal direction ablation | `python -I interp/false_refusal.py` | ~45 min |

`refusal_direction.py` needs `finetune/v2/pools/harm_short_refusals.jsonl` (step 2) for its harmful examples.
Fine-tuned adapters are optional: missing ones are skipped.

## Notes

- **Laptop tips:** `caffeinate -i` keeps the Mac awake but not with the lid closed; keep it plugged in and open.
  Throughput drops sharply when it throttles. A rare `METAL Internal Error` can kill a run: delete that run's
  adapter folder before restarting, because mlx-lm writes `adapters.safetensors` at every checkpoint.
- **Refusal checks** are keyword rules (see each script). They miss some polite refusals and new refusal styles,
  so the article's harmful-answer numbers were read by hand.
- **Safety:** the scripts generate answers to harmful prompts locally. The output folders are git-ignored; don't
  publish raw outputs.
- One small model, mostly one seed reviewed by hand. Treat results as a worked example, not a benchmark.

## References

- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (2024). https://arxiv.org/abs/2406.11717
- Wang, Hu, Röttger, Plank, "Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation" (ICLR 2025). https://arxiv.org/abs/2410.03415
- Röttger et al., "XSTest" (NAACL 2024). https://arxiv.org/abs/2308.01263
- Qi et al., "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!" (ICLR 2024).
- Datasets: WildJailbreak (Allen AI), Tulu 3 SFT mixture (Allen AI), BeaverTails (PKU), Do-Not-Answer (LibrAI), XSTest.
