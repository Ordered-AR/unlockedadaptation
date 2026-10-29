# Unlocking Any-Order Generation in Pretrained Autoregressive Image Models (NeurIPS 2026)

Adapts a pretrained raster-scan [LlamaGen](https://arxiv.org/abs/2406.06525)
checkpoint into an any-order autoregressive image generator, via a short
continued-training phase that adds a future-aware positional embedding and
an ordering curriculum (`--position-order {random, raster, scheduled}`) to
the existing architecture. The adapted model supports sampling under
arbitrary token orders, and inpainting/outpainting/region-conditioned
editing as direct instances of conditional generation, without a
task-specific head or additional retraining. Architecture, tokenizer, and
inference stack (KV caching, parallel decoding, vLLM serving) are otherwise
unchanged from the base model.

Implementation and experiments for *Unlocking Any-Order Generation in
Pretrained Autoregressive Image Models* (NeurIPS 2026); see
[Citation](#citation) below.

## Repo layout

```
autoregressive/
├── models/
│   ├── gpt_position_order.py       # the adapted model: future-aware embedding,
│   │                                # target-position-indexed 2D RoPE routing
│   ├── generate_position_order.py  # any-order prefill + decode loop (KV-cached)
│   ├── generate_nocache.py         # parallel / chunked decoding
│   ├── gpt.py, generate.py         # unmodified base LlamaGen architecture —
│   │                                # what gets adapted, and the base checkpoint's
│   │                                # decode loop used by the --base-model sanity flag
├── train/
│   ├── train_c2i_position_order.py # the adaptation recipe: curriculum sampler +
│   │                                # future-aware embedding, 10x LR for the new table
│   └── extract_codes_c2i.py        # pre-tokenize ImageNet with a frozen VQ tokenizer
├── sample/
│   ├── sample_c2i_position_order.py / _ddp.py  # sample under random/raster/scheduled order
│   ├── sample_c2i_inpaint.py                   # keep tokens outside bbox, regen inside
│   ├── sample_c2i_outpaint.py                  # keep tokens inside bbox, regen outside
│   └── sample_c2i_nocache.py                   # parallel decoding
└── serve/
    ├── gpt_model_position_order_v2.py  # vLLM 0.13.0 custom model
    └── sample_c2i_position_order_v2.py # vLLM-backed sampler (see serve/README.md)

tokenizer/tokenizer_image/  # frozen VQ-16 tokenizer (encode/decode only; not trained here)
dataset/                    # ImageNet — pixels or pre-extracted codes
utils/                      # checkpointing, LR scheduling, config loading, distributed, EMA
evaluations/c2i/            # FID / sFID / IS / Precision / Recall
configs/                    # YAML configs for train_c2i_position_order.py
tests/                      # permutation / RoPE-indexing sanity checks
scripts/                    # example commands for each workflow below
```

## Setup

```bash
pip install -r requirements.txt
```

To adapt a checkpoint yourself, start from a pretrained raster-scan LlamaGen
checkpoint (`FoundationVision/LlamaGen` on HuggingFace: `vq_ds16_c2i.pt` for
the tokenizer, `c2i_{B,L,XL}_{256,384}.pt` for the base GPT).

## Pretrained checkpoints

Already-adapted position-order checkpoints (`gpt_position_order.py`, ready to
use with the sampling/inpainting/outpainting scripts below):

- [orderar/ar-transform-position-order-gpt-l-256](https://huggingface.co/orderar/ar-transform-position-order-gpt-l-256) — GPT-L, 256x256
- [orderar/ar-transform-position-order-gpt-xl-384](https://huggingface.co/orderar/ar-transform-position-order-gpt-xl-384) — GPT-XL, 384x384

Pair either with the base LlamaGen VQ-16 tokenizer (`vq_ds16_c2i.pt`) above.

## Workflow

**1. Pre-tokenize ImageNet** with the frozen VQ tokenizer:
[`scripts/extract_codes.sh`](scripts/extract_codes.sh).

**2. Adapt** a pretrained checkpoint with the ordering curriculum:
[`scripts/train_position_order.sh`](scripts/train_position_order.sh)
→ [`train_c2i_position_order.py`](autoregressive/train/train_c2i_position_order.py).
`--position-order {random, raster, scheduled}` selects the training
distribution over permutations (`scheduled` is the curriculum used in the
paper); see [`configs/`](configs/) for per-size and per-strategy defaults.

**3. Sample** under any order:
[`scripts/sample_position_order.sh`](scripts/sample_position_order.sh).

**4. Inpaint / outpaint** — no retraining, no mask-conditional head:
[`scripts/inpaint.sh`](scripts/inpaint.sh),
[`scripts/outpaint.sh`](scripts/outpaint.sh).
`--bbox lx,ly,rx,ry` is in patch coordinates (0 to `latent_size`, e.g. 0–16 for
256x256 with the VQ-16 tokenizer). Pass `--base-model` to route the same
script through the unmodified pretrained raster model as a sanity check —
inpainting fails on it by construction.

**5. Parallel decoding** — sample multiple tokens per forward pass instead of one:
[`scripts/sample_parallel_decoding.sh`](scripts/sample_parallel_decoding.sh).

**6. Fast serving with vLLM**:
[`scripts/serve_vllm.sh`](scripts/serve_vllm.sh); see
[`autoregressive/serve/README.md`](autoregressive/serve/README.md) for how a
permutation-indexed model is fit into vLLM's decoder contract.

**7. Evaluate** (FID / sFID / IS / Precision / Recall):
[`scripts/eval_fid.sh`](scripts/eval_fid.sh).

## Tests

```bash
python tests/test_generate_position_order.py
```

Checks the inverse-permutation math, identity-order no-op behavior, order-seed
determinism, and that inference-time RoPE indexing matches the training-time
reordering — the invariants the adapted model depends on.


## Citation

```bibtex
@inproceedings{pramanik2026unlocking,
  title     = {Unlocking Any-Order Generation in Pretrained Autoregressive Image Models},
  author    = {Pramanik, Rishav and Pedersoli, Marco and Yin, Zhaozheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
