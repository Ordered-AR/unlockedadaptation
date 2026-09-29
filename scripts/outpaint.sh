#!/bin/bash
# Outpaint: keep tokens inside `--bbox`, regenerate everything outside it.
set -euo pipefail

python autoregressive/sample/sample_c2i_outpaint.py \
    --gpt-model GPT-L \
    --gpt-ckpt /path/to/position_order_checkpoint.pt \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --image-size 256 \
    --bbox 4,4,12,12 \
    --class-label 207 \
    --num-samples 4 \
    --position-order random \
    "$@"
