#!/bin/bash
# Adapt a pretrained raster-scan LlamaGen checkpoint into an any-order
# generator with the ordering curriculum (Section 4.2 of the paper).
set -euo pipefail

torchrun --nproc_per_node=8 autoregressive/train/train_c2i_position_order.py \
    --gpt-model GPT-L \
    --gpt-ckpt ./pretrained/c2i_L_256.pt \
    --code-path /path/to/imagenet_code_c2i \
    --cloud-save-path /path/to/checkpoints \
    --image-size 256 \
    --position-order scheduled \
    --position-order-transition-epochs 100 \
    --epochs 300 \
    --global-batch-size 512 \
    "$@"
