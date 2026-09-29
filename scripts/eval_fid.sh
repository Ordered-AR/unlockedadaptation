#!/bin/bash
# Generate 50k class-conditional samples and score them against the
# ImageNet-256 reference batch (FID / sFID / IS / Precision / Recall).
set -euo pipefail

torchrun --nproc_per_node=8 autoregressive/sample/sample_c2i_position_order_ddp.py \
    --gpt-model GPT-L \
    --gpt-ckpt /path/to/position_order_checkpoint.pt \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --image-size 256 \
    --cfg-scale 1.75 \
    --num-fid-samples 50000 \
    --sample-dir samples \
    "$@"

python evaluations/c2i/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    samples/*.npz
