#!/bin/bash
# v2 sweep: train + XSTest-evaluate every (seed, size) on the shape-matched data. Safe to stop and rerun.
# Every run gets the SAME number of training steps (750 = v1's n1500 run), so data size is the only thing that changes.
# v2 n1500 vs v1 n1500 is then a clean test of the data fix: same size, same steps, different data.
set -eo pipefail   # pipefail: a crash inside "| tee" stops the sweep instead of being skipped
cd "$(dirname "$0")/../.."
source .venv/bin/activate
ITERS=750
for seed in 0 1 2; do
  for n in 1500 5000 10000; do
    ad="finetune/v2/adapters/seed${seed}/n${n}"
    if [ ! -f "$ad/adapters.safetensors" ]; then
      echo "=== training v2 seed${seed} n${n}"
      mlx_lm.lora --model Qwen/Qwen2.5-1.5B-Instruct --train --data "finetune/v2/data/seed${seed}/n${n}" \
        --fine-tune-type lora --num-layers 16 --batch-size 4 --iters $ITERS --learning-rate 1e-5 \
        --mask-prompt --steps-per-eval 250 --max-seq-length 1024 --grad-checkpoint \
        --seed "$seed" --adapter-path "$ad" 2>&1 | tee "finetune/v2/logs_seed${seed}_n${n}.txt"
    fi
  done
done
python -I finetune/eval_xstest_sweep.py --root finetune/v2/adapters --sizes 1500,5000,10000 --out xstest_v2 --no-base
echo "V2 SWEEP DONE"
