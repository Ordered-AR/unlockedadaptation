# No-cache parallel decoding for Position Order GPT.
#
# Each step runs a FULL forward pass over the partial sequence — no KV cache,
# no risk of stale entries. To get parallelism, we append `chunk_k - 1`
# placeholder positions at the end and read `chunk_k` logits in one pass:
#
#     [cls, t_0, ..., t_{n-1}, ph_0, ph_1, ..., ph_{k-2}]
#       └── prefix (real) ────┘└── placeholders ────────┘
#
# Training convention: position cls+i has tok_emb(token_at_i) + target_pos_emb(step i+1)
# and its logit predicts step i+1. So:
#   - position cls+n-1 (last real token) naturally predicts step n
#   - placeholder at position cls+n+j-1 (j=1..k-1) gets target_pos_emb(step n+j)
#     and its logit predicts step n+j
#
# Mask: INDEPENDENT — each chunk position attends to prefix + itself only,
# never to other chunk positions. This guarantees the k logits are independent
# predictions given the same prefix context. Quality depends on whether the
# model tolerates these placeholder positions, which it never saw in training.
import math
from typing import Optional, List

import torch
import torch.nn.functional as F


# ─────────────────────────── sampling helpers ────────────────────────────────

def _top_k_top_p(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        kth = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, filter_value)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        idx_remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(idx_remove, filter_value)
    return logits


def _sample_batch(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True):
    """[bsz, k, vocab] → [bsz, k] sampled tokens."""
    bsz, k, vocab = logits.shape
    logits = logits / max(temperature, 1e-5)
    flat = logits.reshape(bsz * k, vocab)
    if top_k > 0 or top_p < 1.0:
        flat = _top_k_top_p(flat, top_k=top_k, top_p=top_p)
    probs = F.softmax(flat, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
        idx = idx.squeeze(1)
    return idx.view(bsz, k)


# ─────────────────────────── chunk schedule ──────────────────────────────────

def make_schedule(num_tokens: int, num_steps: int, schedule: str = "cosine") -> List[int]:
    """Chunk sizes summing to num_tokens with len ≈ num_steps. First chunk = 1."""
    if num_steps >= num_tokens:
        return [1] * num_tokens
    num_steps = max(1, num_steps)

    if schedule == "linear":
        weights = [i + 1 for i in range(num_steps)]
    elif schedule == "cosine":
        if num_steps == 1:
            weights = [1.0]
        else:
            weights = [
                0.05 + 0.95 * (1 - math.cos(math.pi * i / (num_steps - 1))) / 2
                for i in range(num_steps)
            ]
    elif schedule == "exponential":
        weights = [2 ** i for i in range(num_steps)]
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    total = sum(weights)
    raw = [w * num_tokens / total for w in weights]
    chunks = [max(1, int(round(r))) for r in raw]
    chunks[0] = 1

    diff = num_tokens - sum(chunks)
    i = len(chunks) - 1
    while diff > 0:
        chunks[i] += 1
        diff -= 1
        i = (i - 1) if i > 0 else (len(chunks) - 1)
    while diff < 0 and any(c > 1 for c in chunks):
        i = max(range(len(chunks)), key=lambda j: chunks[j])
        chunks[i] -= 1
        diff += 1

    return chunks


# ───────────────────── core no-cache forward ──────────────────────────────────

@torch.no_grad()
def _nocache_forward(
    model,
    cond_idx: torch.Tensor,
    generated: torch.Tensor,        # [bsz, n] in token_order sequence
    token_order: torch.Tensor,      # [bsz, block_size]
    chunk_k: int,
    placeholder_id: int = 0,
    independent_chunk: bool = True,
):
    """One forward pass returning chunk_k logits for steps n..n+chunk_k-1.

    Sequence layout (length = cls + n + chunk_k - 1 when chunk_k>=1, n>=0):
        [cls × cls_token_num, t_0..t_{n-1}, ph_0..ph_{chunk_k-2}]

    target_pos_embed at each position (training convention: position predicts step+1):
        cls positions:           token_order[:, 0]      (predict step 0)
        real tokens (i=0..n-1):  token_order[:, i+1]    (predict step i+1)
        placeholders (j=0..k-2): token_order[:, n+j+1]  (predict step n+j+1)

    Logits at positions cls+n-1..cls+n+chunk_k-2 give predictions for steps n..n+chunk_k-1.
    Special case n=0: position cls-1 (last cls) gives step 0.

    Mask:
      - independent_chunk=True (default): chunk positions see ONLY the prefix
        (real tokens and cls), not each other or themselves through cross-attention
        to other chunks. They DO see themselves to avoid degenerate attention.
      - independent_chunk=False: standard causal — placeholders attend to prior
        placeholders too (fused sequential — chunk tokens are not independent).
    """
    bsz = generated.shape[0]
    n = generated.shape[1]
    k = chunk_k
    device = generated.device
    cls_n = model.cls_token_num
    n_img = n + max(0, k - 1)  # number of image-like positions (real + placeholder)

    # ── 1. Build embeddings ─────────────────────────────────────────────────
    cond_emb = model.cls_embedding(cond_idx, train=False)[:, :cls_n]
    parts = [cond_emb]
    if n > 0:
        parts.append(model.tok_embeddings(generated))
    if k > 1:
        ph_ids = torch.full((bsz, k - 1), placeholder_id, dtype=torch.long, device=device)
        parts.append(model.tok_embeddings(ph_ids))
    tok_emb = torch.cat(parts, dim=1)  # [bsz, cls + n_img, dim]

    # target_pos_embed: position i in image stream predicts step i+1
    cls_target = token_order[:, :1].expand(-1, cls_n)  # [bsz, cls]
    if n_img > 0:
        # positions 0..n_img-1 in image stream predict steps 1..n_img
        img_target = token_order[:, 1 : n_img + 1]
        all_target = torch.cat([cls_target, img_target], dim=1)
    else:
        all_target = cls_target
    target_emb = model.target_pos_embed(all_target)

    h = tok_emb + target_emb
    L = h.shape[1]
    h = model.tok_dropout(h)
    model.freqs_cis = model.freqs_cis.to(device)

    # ── 2. Order-aware freqs_cis ────────────────────────────────────────────
    cls_freqs = model.freqs_cis[:cls_n].unsqueeze(0).expand(bsz, -1, -1, -1)
    if n_img > 0:
        spatial = token_order[:, :n_img]
        img_freqs = model.freqs_cis[cls_n + spatial]
        freqs_cis = torch.cat([cls_freqs, img_freqs], dim=1)
    else:
        freqs_cis = cls_freqs

    # ── 3. Mask ─────────────────────────────────────────────────────────────
    prefix_len = cls_n + n  # [0..prefix_len-1] = real prefix
    chunk_start = max(prefix_len, cls_n)  # where placeholders begin (only relevant if k>1)
    # (When n==0, the "prefix" is just cls; placeholders start at cls_n, but we still
    # treat position cls_n-1 as the "predict step 0" position which is valid.)

    # Standard causal as base
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

    if independent_chunk and k > 1:
        # Chunk positions are at indices cls_n+n..cls_n+n+k-2 (i.e. cls_n+n .. L-1).
        # They should see ONLY the prefix (positions 0..cls_n+n-1) and themselves.
        # Zero out chunk-to-chunk attention.
        for i in range(prefix_len, L):
            for j in range(prefix_len, L):
                if i != j:
                    mask[i, j] = False
        # Note: position cls_n+n-1 (last real token) is in the prefix; its row
        # is unchanged. Its logit predicts step n.

    mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1)
    input_pos = torch.arange(L, device=device)

    # ── 4. Run blocks with KV cache disabled ────────────────────────────────
    saved = []
    for layer in model.layers:
        saved.append(layer.attention.kv_cache)
        layer.attention.kv_cache = None
    try:
        for layer in model.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        h = model.norm(h)
        logits = model.output(h).float()
    finally:
        for layer, c in zip(model.layers, saved):
            layer.attention.kv_cache = c

    # ── 5. Read k logits ────────────────────────────────────────────────────
    # When n>=1: start at position cls_n+n-1 (last real token); k consecutive logits.
    # When n==0: start at position cls_n-1 (last cls); k consecutive logits.
    start = (cls_n - 1) if n == 0 else (cls_n + n - 1)
    return logits[:, start : start + k, :]  # [bsz, k, vocab]


# ───────────────────────── public entry point ────────────────────────────────

@torch.no_grad()
def generate_nocache(
    model,
    cond,
    max_new_tokens: int,
    token_order: Optional[torch.Tensor] = None,
    num_steps: int = 64,
    schedule: str = "cosine",
    independent_chunk: bool = True,
    placeholder_id: int = 0,
    cfg_scale: float = 1.0,
    cfg_interval: int = -1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits: bool = True,
):
    """No-cache parallel generation.

    Args:
        num_steps: total forward passes. num_steps=max_new_tokens → fully sequential.
        schedule:  "linear" | "cosine" | "exponential" — chunk size growth.
        independent_chunk:
            True  → chunk positions see only the prefix (true parallel, untrained pattern)
            False → causal mask (placeholders attend to prior placeholders; fused seq)
        placeholder_id: token id for placeholder positions (default 0).
    """
    bsz = cond.shape[0]
    device = cond.device

    if token_order is None:
        token_order = torch.stack([
            torch.randperm(max_new_tokens, device=device) for _ in range(bsz)
        ])
    assert token_order.shape == (bsz, max_new_tokens)

    chunks = make_schedule(max_new_tokens, num_steps, schedule=schedule)

    # CFG: prepare unconditional batch
    if cfg_scale > 1.0:
        if model.model_type == "c2i":
            cond_null = torch.ones_like(cond) * model.num_classes
        elif model.model_type == "t2i":
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
        else:
            raise ValueError(f"unsupported model_type: {model.model_type}")
        cond_combined = torch.cat([cond, cond_null], dim=0)
        token_order_combined = torch.cat([token_order, token_order], dim=0)
    else:
        cond_combined = cond
        token_order_combined = token_order

    generated = torch.zeros(bsz, 0, dtype=torch.long, device=device)
    pos_idx = 0
    cfg_flag = True

    for chunk_k in chunks:
        if cfg_interval > -1 and pos_idx > cfg_interval:
            cfg_flag = False

        if cfg_scale > 1.0:
            gen_combined = torch.cat([generated, generated], dim=0)
            logits = _nocache_forward(
                model, cond_combined, gen_combined, token_order_combined, chunk_k,
                placeholder_id=placeholder_id, independent_chunk=independent_chunk,
            )
            cond_logits, uncond_logits = torch.split(logits, bsz, dim=0)
            if cfg_flag:
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits = cond_logits
        else:
            logits = _nocache_forward(
                model, cond, generated, token_order, chunk_k,
                placeholder_id=placeholder_id, independent_chunk=independent_chunk,
            )

        new_tokens = _sample_batch(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            sample_logits=sample_logits,
        )
        generated = torch.cat([generated, new_tokens], dim=1)
        pos_idx += chunk_k

    # Un-permute to raster
    inv_order = torch.zeros_like(token_order)
    inv_order.scatter_(
        1, token_order,
        torch.arange(max_new_tokens, device=device).unsqueeze(0).expand_as(token_order),
    )
    return torch.gather(generated, 1, inv_order)
