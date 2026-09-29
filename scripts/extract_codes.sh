#!/bin/bash
# Pre-tokenize ImageNet into VQ codes, once, before training.
set -euo pipefail

torchrun --nproc_per_node=8 autoregressive/train/extract_codes_c2i.py \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --data-path /path/to/imagenet/train \
    --code-path /path/to/imagenet_code_c2i \
    --image-size 256 \
    "$@"
