#!/bin/bash
# Sample from a trained position-order model; visualizes the sampled
# generation order as a heatmap alongside the result.
set -euo pipefail

python autoregressive/sample/sample_c2i_position_order.py \
    --gpt-model GPT-L \
    --gpt-ckpt /path/to/position_order_checkpoint.pt \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --image-size 256 \
    --position-order random \
    --cfg-scale 1.75 \
    "$@"
