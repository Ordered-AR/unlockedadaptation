#!/bin/bash
# Inpaint: feed known tokens outside `--bbox` as context, regenerate the
# masked interior autoregressively in the chosen order. No masking head,
# no retraining — a direct instance of any-order conditional generation.
set -euo pipefail

python autoregressive/sample/sample_c2i_inpaint.py \
    --gpt-model GPT-L \
    --gpt-ckpt /path/to/position_order_checkpoint.pt \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --image-size 256 \
    --bbox 4,4,12,12 \
    --class-label 207 \
    --num-samples 4 \
    --position-order random \
    "$@"

# Sanity check: force the pretrained raster model through the same script
# (fails to inpaint coherently by construction):
#   python autoregressive/sample/sample_c2i_inpaint.py --base-model \
#       --gpt-ckpt ./pretrained/c2i_L_256.pt ...
