#!/bin/bash
# Fast sampling via vLLM's paged-attention engine (see autoregressive/serve/README.md
# for how a permutation-indexed model is made to fit vLLM's decoder contract).
set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python autoregressive/serve/sample_c2i_position_order_v2.py \
    --gpt-model GPT-L \
    --gpt-ckpt /path/to/position_order_checkpoint.pt \
    --vq-ckpt ./pretrained/vq_ds16_c2i.pt \
    --image-size 256 --downsample-size 16 \
    --position-order random --order-seed 42 \
    --cfg-scale 1.75 --temperature 1.0 --top-k 2000 --seed 0 \
    "$@"
