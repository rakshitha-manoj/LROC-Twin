#!/bin/bash
# run_rank_sweep.sh
# Runs all three LoRA rank configurations sequentially.
# Each run saves its adapter to outputs/lroc-twin-r{rank}/final_adapter/
# Estimated total time on RTX 3050 4GB: 3 to 5 hours.

set -e

echo "=== LROC-Twin LoRA Rank Sweep ==="
echo "Configs: r=8, r=16, r=32"
echo ""

python training/scripts/train.py --config training/configs/lora_r8.yaml
echo "r=8 complete."

python training/scripts/train.py --config training/configs/lora_r16.yaml
echo "r=16 complete."

python training/scripts/train.py --config training/configs/lora_r32.yaml
echo "r=32 complete."

echo ""
echo "Rank sweep done. Run eval/in_domain/compare.py to evaluate all adapters."
