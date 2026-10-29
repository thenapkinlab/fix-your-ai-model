#!/bin/bash
# Train + evaluate every (seed, size). Safe to stop and rerun: finished runs are skipped.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
for seed in 0 1 2; do
  for n in 250 500 1000 1500; do
    name="seed${seed}_n${n}"
    ad="finetune/adapters/seed${seed}/n${n}"
    if [ ! -f "$ad/adapters.safetensors" ]; then
      echo "=== training $name"
      mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data "finetune/data/seed${seed}/n${n}" \
        --fine-tune-type lora --num-layers 16 --batch-size 4 --iters $(( n * 2 / 4 )) --learning-rate 1e-5 \
        --mask-prompt --steps-per-eval 50 --val-batches -1 --max-seq-length 1024 --grad-checkpoint \
        --seed "$seed" --adapter-path "$ad" 2>&1 | tee "finetune/logs_${name}.txt"
    fi
    if [ ! -f "finetune/results/${name}.json" ]; then
      echo "=== evaluating $name"
      python -I finetune/eval_heldout.py --name "$name" --adapter "$ad"
    fi
  done
done
echo "SWEEP DONE"
