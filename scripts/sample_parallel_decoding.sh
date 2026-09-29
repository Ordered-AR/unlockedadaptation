#!/bin/bash
# Parallel decoding: any-order training makes tokens with near-independent
# conditionals samplable jointly in one forward pass. --num-steps 64 means
# 256 image tokens are produced in 64 forward passes (4x fewer than the
# fully sequential 256), at no measurable FID cost (Section 5.3).
set -euo pipefail

python autoregressive/sample/sample_c2i_nocache.py \
    --gpt-model GPT-L \
    --gpt-ckpt /path/to/position_order_checkpoint.pt \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --image-size 256 \
    --num-steps 64 \
    --schedule cosine \
    --cfg-scale 1.75 \
    "$@"
