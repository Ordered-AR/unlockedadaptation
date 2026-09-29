# Serving with vLLM — Position-Order AR Generation

This directory adapts the position-order class-conditional image
generator to run on **vLLM 0.13.0**, giving a large sampling speedup
over the PyTorch reference loop in
[../sample/sample_c2i_position_order.py](../sample/sample_c2i_position_order.py).

The core trick is making vLLM treat a non-standard model — one whose
attention positions are a *permutation* of the output positions, not a
running counter — as a regular decoder. We do that by registering a
custom `TransformerPositionOrder` model and injecting a `token_order`
tensor onto it before each `llm.generate(...)` call.

---

## Files

| File | Purpose |
|---|---|
| [gpt_model_position_order_v2.py](gpt_model_position_order_v2.py) | vLLM 0.13.0 custom model (`TransformerPositionOrder`) with 2D spatial RoPE, `target_pos_embed`, and CFG-aware `compute_logits` |
| [sample_c2i_position_order_v2.py](sample_c2i_position_order_v2.py) | Launcher: builds a temp config dir, boots `vllm.LLM`, injects token_order + checkpoint weights, samples, un-permutes |
| [fake_json/GPT-{B,L,XL}-PO/config.json](fake_json/) | HF-style config stubs required by vLLM model-loading — consumed at init, `block_size` is overwritten at runtime to match `image_size / downsample_size` |

---

## Quick start

```bash
pip install vllm==0.13.0
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
```

```bash
python autoregressive/serve/sample_c2i_position_order_v2.py \
    --gpt-model GPT-L \
    --gpt-ckpt  /path/to/model.pt \
    --vq-ckpt   /path/to/vq_ds16_c2i.pt \
    --image-size 256 --downsample-size 16 \
    --position-order random --order-seed 42 \
    --cfg-scale 1.75 --temperature 1.0 --top-k 2000 --seed 0
```

Output: `sample_c2i_position_order_vllm_v2.png` (4×2 grid by default,
using the class list hard-coded in
[sample_c2i_position_order_v2.py](sample_c2i_position_order_v2.py#L94)).

---

## How it fits together

```
    class labels [N]                                 ┌─────────────┐
        │                                            │ VQ decoder  │
        │                                            │ (pytorch)   │
        ▼                                            └──────▲──────┘
    ┌───────────────┐   token_order (CPU→CUDA)              │ index_sample
    │ build prompt  │──────────────────────────┐            │ (raster order)
    │ [cls_idx] ×N  │                          │            │
    └───────┬───────┘                          ▼            │
            │                         ┌────────────────┐    │
            ▼                         │ apply_model:   │    │
    (CFG double: +null class cond)    │  - load ckpt   │    │
            │                         │  - set         │    │
            │                         │    token_order │    │
            ▼                         │  - set cond_bsz│    │
    ┌───────────────┐                 └────────┬───────┘    │
    │   vllm.LLM    │◄─── register_model ◄────┘            │
    │   (0.13.0)    │     TransformerPositionOrder          │
    │               │                                       │
    │ .generate()   │── token_ids in step order ───► invert permutation ─┘
    └───────────────┘                                 (scatter along dim=1)
```

---

## Custom model contract

vLLM 0.13.0 expects a nn.Module implementing:

```python
class TransformerPositionOrder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, input_ids, positions,
                intermediate_tensors=None, inputs_embeds=None): ...
    def compute_logits(self, hidden_states): ...
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]: ...
```

Spec reproduced at the top of
[gpt_model_position_order_v2.py](gpt_model_position_order_v2.py#L1).

### How `token_order` is used every step

- The caller injects a `[B, block_size]` long tensor once, before
  generation starts.
- `forward` derives the current decode step from `positions`:

  ```python
  is_prefill  = positions.max() == 0
  decode_step = positions[0].item() - cls_token_num
  ```

  This relies on vLLM's guarantee that a homogeneous batch advances in
  lockstep, so every sequence is at the same step at any engine tick.
- `spatial_pos = token_order[:, decode_step]` selects:
  - the RoPE `freqs_cis` slice for this step, and
  - the `target_pos_embed` index to add to the input embedding (so the
    model knows *which* spatial slot it is about to emit).

### CFG in `compute_logits`

The batch is doubled: rows `0..N-1` are conditioned on the class label,
rows `N..2N-1` on the null class. `compute_logits` combines them with

```
logit = (1 + cfg_scale) * cond - cfg_scale * uncond
```

and returns the same combined logit for both halves so vLLM samples the
same token for each pair, keeping them in sync for the next step.
`cond_bsz = N` is injected alongside `token_order`.

---

## Invariants / gotchas

**`load_format="dummy"` + `LLM.apply_model()`.**  vLLM wants weight
files at init. We hand it random weights and load the real checkpoint
*afterwards* inside the worker via `llm.apply_model(...)`. This avoids
having to repackage checkpoints into HF-shard form.

**`freqs_cis` is a plain attribute, not a buffer.**  If it were a
`register_buffer`, vLLM's dummy-init pass would zero it out. It is
precomputed on CPU at init and moved to CUDA lazily on first forward.

**Closure serialization.**  The vLLM v1 engine runs the model in a
*subprocess*. `apply_model(fn)` pickles `fn` and sends it over IPC.
The default vLLM encoder cannot handle closures, so we set
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` to fall back to pickle.

**CPU tensors in closures.**  The closure that sets `token_order` must
hold a **CPU** tensor and move it to CUDA *inside* the worker:

```python
token_orders_cpu = token_orders_full.cpu()
llm.apply_model(lambda m: setattr(m, "token_order", token_orders_cpu.cuda()))
```

Passing a CUDA tensor through IPC silently corrupts it (it's a handle
into the parent's CUDA context, not the worker's).

**`enforce_eager=True`.**  `token_order` is dynamic per-call, so CUDA
graphs would capture stale values. Eager mode is mandatory.

**`ignore_eos=True`.**  The fake config sets `eos_token_id=2`, but
token 2 is a valid VQ codebook token. Without `ignore_eos`, generation
truncates whenever the sampler happens to pick token 2.

**`block_size` is patched at runtime.**  The shipped
`fake_json/*-PO/config.json` stubs hard-code one resolution. The
launcher copies the config to a temp dir and rewrites `block_size` to
`(image_size / downsample_size) ** 2` so it matches the checkpoint.

**Un-permutation happens on the host.**  vLLM returns tokens in
**generation step** order. Reorder to raster order before the VQ
decoder:

```python
index_sample = torch.zeros_like(generated)
index_sample.scatter_(1, token_order_gpu, generated)
```

---

## Performance

- `enforce_eager=True`, `enable_prefix_caching=False`,
  `enable_chunked_prefill=False` — all set in
  [sample_c2i_position_order_v2.py](sample_c2i_position_order_v2.py#L139).
  Prefix caching is useless here (prompts are single class tokens); the
  other two simplify scheduling for a homogeneous image-generation batch.
- `dtype=bfloat16`, `gpu_memory_utilization=0.9`.
- Batch is effectively `2N` when CFG is on; vLLM schedules them together.

---

## Debugging

If the vLLM model forward pass diverges from the reference PyTorch
model (`sample_c2i_position_order.py`), the usual suspects are:

- `freqs_cis` clobbered by dummy init (see *plain attribute* note)
- `token_order` tensor on the wrong device / stale
- `decode_step` off-by-one (forgot to subtract `cls_token_num`)
- CFG combination applied twice (once in `compute_logits`, once in the
  sampling loop)
