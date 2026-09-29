# vLLM 0.13.0-compatible model for position-order AR image generation.
#
# Interface contract (vLLM 0.13.0 custom model):
#   __init__(self, *, vllm_config: VllmConfig, prefix: str = "")
#   forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None)
#     → hidden_states [total_tokens, dim]
#   compute_logits(hidden_states) → logits [total_tokens, vocab_size]
#   load_weights(weights: Iterable[tuple[str, Tensor]]) → set[str]
#
# Token-order injection:
#   Before calling LLM.generate(), the caller sets model.token_order via:
#     llm.apply_model(lambda m: setattr(m, "token_order", token_orders_full.cuda()))
#   token_order shape: [batch_size, block_size], dtype=torch.long (GPU tensor)
#   All sequences in a batch must be at the same decode step (standard vLLM behaviour
#   for a homogeneous batch started together).
#
# decode_step is derived from the positions tensor:
#   is_prefill  = positions.max() == 0
#   decode_step = positions[0].item() - cls_token_num   (0-indexed image-token step)

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    dim: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5

    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    model_type: str = "c2i"
    cfg_scale: float = 4.0

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_seq_len: int = 2048

    # Position-order specific
    position_order: str = "random"


def _find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


# ---------------------------------------------------------------------------
# RoPE utilities (2-D spatial)
# ---------------------------------------------------------------------------

def precompute_freqs_cis_2d(
    grid_size: int,
    n_elem: int,
    base: int = 10000,
    cls_token_num: int = 1,
) -> torch.Tensor:
    """Return [cls_token_num + grid_size**2, n_elem//2, 2] freqs_cis buffer.

    Index 0..cls_token_num-1: zeros (no rotation for class tokens).
    Index cls_token_num + k:  2-D spatial RoPE for flattened grid position k.
    """
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, half_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, half_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, half_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)  # (grid_size**2, half_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num + grid_size**2, half_dim // 2, 2)
    return cond_cache


def apply_rotary_emb_flat(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to a flat token batch.

    Args:
        x:         [N, n_head, head_dim]
        freqs_cis: [N, head_dim//2, 2]   (per-token, per-sample spatial positions)
    Returns:
        [N, n_head, head_dim]
    """
    x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # x_shaped: [N, n_head, head_dim//2, 2]
    fc = freqs_cis.unsqueeze(1)  # [N, 1, head_dim//2, 2]
    x_out = torch.stack(
        [
            x_shaped[..., 0] * fc[..., 0] - x_shaped[..., 1] * fc[..., 1],
            x_shaped[..., 1] * fc[..., 0] + x_shaped[..., 0] * fc[..., 1],
        ],
        dim=-1,
    ).flatten(2)  # [N, n_head, head_dim]
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Model sub-modules
# ---------------------------------------------------------------------------

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = _find_multiple(hidden_dim, config.multiple_of)

        self.w_merged = nn.Linear(config.dim, hidden_dim * 2, bias=False)
        self.act_fn = SiluAndMul()
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w_merged(x)))


class AttentionPO(nn.Module):
    """Attention layer with per-sample spatial 2-D RoPE for position-order generation.

    Unlike standard attention, freqs_cis is computed outside (in the Transformer
    forward) and passed in, so each token in the batch can have a different
    spatial position.
    """

    def __init__(
        self,
        config: ModelArgs,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        # head_dim=100 needs padding; use the same monkey-patch as gpt_model.py
        self._need_pad = (self.head_dim == 100)
        attn_head_size = 112 if self._need_pad else self.head_dim
        attn_scale = 100 ** -0.5 if self._need_pad else self.head_dim ** -0.5

        self.attn = Attention(
            num_heads=self.n_head,
            head_size=attn_head_size,
            scale=attn_scale,
            num_kv_heads=self.n_kv_head,
            cache_config=vllm_config.cache_config,
            prefix=maybe_prefix(prefix, "attn"),
        )

    def forward(
        self,
        x: torch.Tensor,            # [N, dim]  — flat batch, one token per sequence
        freqs_cis: torch.Tensor,    # [N, head_dim//2, 2]  — per-sample spatial pos
    ) -> torch.Tensor:
        N = x.shape[0]
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(N, self.n_head, self.head_dim)
        xk = xk.view(N, self.n_kv_head, self.head_dim)

        xq = apply_rotary_emb_flat(xq, freqs_cis)  # [N, n_head, head_dim]
        xk = apply_rotary_emb_flat(xk, freqs_cis)  # [N, n_kv_head, head_dim]

        if self._need_pad:
            # GPT-3B has head_dim=100; vLLM paged-attn requires supported sizes
            pad = torch.zeros(N, self.n_head, 12, device=x.device, dtype=x.dtype)
            xq = torch.cat([xq, pad], dim=-1)
            pad_kv = torch.zeros(N, self.n_kv_head, 12, device=x.device, dtype=x.dtype)
            xk = torch.cat([xk, pad_kv], dim=-1)
            xv = xv.view(N, self.n_kv_head, self.head_dim)
            xv = torch.cat([xv, pad_kv], dim=-1)
            xq = xq.flatten(1)
            xk = xk.flatten(1)
            xv = xv.flatten(1)

            output = self.attn(xq, xk, xv)  # [N, n_head * 112]
            output = output.view(N, self.n_head, 112)[..., :100].flatten(1)
        else:
            xq = xq.flatten(1)  # [N, n_head * head_dim]
            xk = xk.flatten(1)
            # xv stays [N, n_kv_head * head_dim]
            output = self.attn(xq, xk, xv)  # [N, n_head * head_dim]

        return self.wo(output)


class TransformerBlockPO(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attention = AttentionPO(config, vllm_config, prefix=maybe_prefix(prefix, "attention"))
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,         # [N, dim]
        freqs_cis: torch.Tensor, # [N, head_dim//2, 2]
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


# ---------------------------------------------------------------------------
# Top-level model (vLLM 0.13.0 interface)
# ---------------------------------------------------------------------------

class TransformerPositionOrder(nn.Module):
    """Position-order GPT image-generation model for vLLM 0.13.0.

    Token-order injection
    ---------------------
    Before the first call to generate(), set::

        llm.apply_model(lambda m: setattr(m, "token_order", orders.cuda()))

    where ``orders`` is [batch_size, block_size] long tensor on GPU.
    All sequences in the batch must start together (same decode step at each
    engine step — the normal vLLM guarantee for a homogeneous batch).

    Model input/output shapes (vLLM 0.13.0 flat format)
    -----------------------------------------------------
    Prefill: input_ids=[N], positions=[N] all zeros.
    Decode k: input_ids=[N], positions=[N] all = cls_token_num + k.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_cfg = vllm_config.model_config.hf_config

        # ---- Build ModelArgs from HF config (OPT-style JSON fields) ----
        config = ModelArgs(
            dim=hf_cfg.hidden_size,
            n_layer=hf_cfg.num_hidden_layers,
            n_head=hf_cfg.num_attention_heads,
            n_kv_head=getattr(hf_cfg, "num_key_value_heads", None),
            vocab_size=hf_cfg.vocab_size,
            num_classes=getattr(hf_cfg, "num_classes", 1000),
            cls_token_num=getattr(hf_cfg, "cls_token_num", 1),
            block_size=getattr(hf_cfg, "block_size", 256),
            cfg_scale=getattr(hf_cfg, "cfg_scale", 4.0),
            model_type=getattr(hf_cfg, "ar_model_type", "c2i"),
            norm_eps=getattr(hf_cfg, "layer_norm_eps", 1e-5),
        )
        self.config = config
        self.vocab_size = config.vocab_size
        self.cls_token_num = config.cls_token_num
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.cfg_scale = config.cfg_scale
        self.model_type = config.model_type

        # ---- Embeddings ----
        if config.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(
                config.num_classes, config.dim, config.class_dropout_prob
            )
        else:
            raise ValueError("vllm serving only supports c2i model_type")

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        # +1 sentinel index for "no next position" at the final decode step
        self.target_pos_embed = nn.Embedding(config.block_size + 1, config.dim)

        # ---- Transformer blocks ----
        self.layers = nn.ModuleList([
            TransformerBlockPO(
                config,
                vllm_config,
                prefix=maybe_prefix(prefix, f"layers.{i}"),
            )
            for i in range(config.n_layer)
        ])

        # ---- Output ----
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # ---- Output head + logits processor ----
        # ParallelLMHead is vLLM's VocabParallelEmbedding-based lm_head;
        # required by LogitsProcessor._get_logits (which calls quant_method.apply).
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.dim,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # ---- 2-D spatial RoPE buffer ----
        # Stored as a plain attribute (NOT register_buffer) so vLLM's dummy-init
        # random weight pass cannot overwrite the computed cos/sin values.
        grid_size = int(config.block_size ** 0.5)
        assert grid_size * grid_size == config.block_size, \
            f"block_size={config.block_size} must be a perfect square"
        self._freqs_cis_cpu = precompute_freqs_cis_2d(
            grid_size, config.dim // config.n_head,
            config.rope_base, config.cls_token_num,
        ).float()  # always float32; shape [cls_token_num + block_size, head_dim//2, 2]
        # Will be moved to GPU in load_weights() after the engine is set up.
        self.freqs_cis: Optional[torch.Tensor] = None

        # ---- Per-batch token order (set externally before generate()) ----
        # shape: [batch_size, block_size], long, on GPU
        self.token_order: Optional[torch.Tensor] = None

        # ---- CFG batch size (set externally before generate()) ----
        # Number of *conditioned* sequences; total batch = 2 * cond_bsz when CFG active.
        # Set via: llm.apply_model(lambda m: setattr(m, "cond_bsz", bsz))
        self.cond_bsz: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_freqs_cis(
        self, positions: torch.Tensor, is_prefill: bool, decode_step: int
    ) -> torch.Tensor:
        """Compute per-sample spatial freqs_cis for the current step.

        Returns [N, head_dim//2, 2] float32 where N = positions.shape[0].
        Always float32 regardless of model dtype: RoPE precision matters.
        """
        N = positions.shape[0]
        if is_prefill:
            # Class tokens: zero RoPE (same as cls index 0 in freqs_cis)
            fc = self.freqs_cis if self.freqs_cis is not None else self._freqs_cis_cpu.to(positions.device)
            return fc[0:1].float().expand(N, -1, -1)  # [N, *, 2]

        # Decode step k: spatial position = token_order[i, k] for each sample i
        if self.token_order is None:
            # Fall back to raster order (for profiling/warmup runs)
            spatial_pos = torch.full((N,), decode_step, dtype=torch.long, device=positions.device)
        else:
            token_order_batch = self.token_order[:N]  # handles CFG (N=2*orig_bsz)
            spatial_pos = token_order_batch[:, decode_step]  # [N]

        idx = self.cls_token_num + spatial_pos  # [N]
        fc = self.freqs_cis if self.freqs_cis is not None else self._freqs_cis_cpu.to(positions.device)
        return fc[idx].float()  # [N, head_dim//2, 2]

    def _build_target_pos(
        self, N: int, is_prefill: bool, decode_step: int, device: torch.device
    ) -> torch.Tensor:
        """Return [N] target spatial positions for target_pos_embed."""
        if is_prefill:
            # Predict step 0: first spatial position per sample
            if self.token_order is None:
                return torch.zeros(N, dtype=torch.long, device=device)
            return self.token_order[:N, 0]  # [N]

        next_step = decode_step + 1
        if next_step >= self.block_size:
            # Sentinel: no next position (last image token step)
            return torch.full((N,), self.block_size, dtype=torch.long, device=device)

        if self.token_order is None:
            return torch.full((N,), next_step, dtype=torch.long, device=device)
        return self.token_order[:N, next_step]  # [N]

    # ------------------------------------------------------------------
    # vLLM 0.13.0 interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                          # [N]
        positions: torch.Tensor,                          # [N]
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = input_ids.shape[0]
        device = input_ids.device

        # Determine phase
        is_prefill = positions.max().item() == 0
        if not is_prefill:
            # Clamp to [0, block_size-1] so profiling/warmup runs with large
            # position values don't trigger out-of-bounds freqs_cis indexing.
            decode_step = int(
                max(0, min(positions[0].item() - self.cls_token_num, self.block_size - 1))
            )
        else:
            decode_step = 0

        # ---- Token embeddings ----
        if is_prefill:
            cond_ids = torch.clamp(input_ids, max=self.num_classes)
            token_emb = self.cls_embedding(cond_ids)  # [N, dim]
        else:
            token_emb = self.tok_embeddings(input_ids)  # [N, dim]

        # ---- Add target_pos_embed (next spatial position signal) ----
        target_pos = self._build_target_pos(N, is_prefill, decode_step, device)
        token_emb = token_emb + self.target_pos_embed(target_pos)  # [N, dim]

        # ---- Spatial RoPE freqs for this step ----
        freqs_cis = self._build_freqs_cis(positions, is_prefill, decode_step)
        # [N, head_dim//2, 2]

        # ---- Transformer blocks ----
        hh = token_emb
        for layer in self.layers:
            hh = layer(hh, freqs_cis)

        return self.norm(hh)  # [N, dim]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's VllmModel Protocol for text-generation runner detection."""
        return self.tok_embeddings(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        N = hidden_states.shape[0]
        if self.cfg_scale > 1.0 and self.cond_bsz is not None and N == 2 * self.cond_bsz:
            # Batch layout: [cond_0..cond_{bsz-1}, uncond_0..uncond_{bsz-1}]
            # CFG: logit = uncond + cfg_scale * (cond - uncond)
            bsz = self.cond_bsz
            raw = self.logits_processor(self.lm_head, hidden_states)  # [2*bsz, vocab_size]
            cond_logits   = raw[:bsz]
            uncond_logits = raw[bsz:]
            cfg_logits = uncond_logits + self.cfg_scale * (cond_logits - uncond_logits)
            # Feed cfg_logits to BOTH halves so the uncond KV cache evolves with the
            # same token distribution as the cond sequences.  This matches the reference
            # implementation (generate_position_order.py) which always feeds the
            # CFG-guided token back to both the cond and uncond branches.
            return torch.cat([cfg_logits, cfg_logits], dim=0)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights from the AR-Transform checkpoint format.

        Differences from standard state_dict:
          - w1 + w3 per FFN layer → merged into w_merged
          - freqs_cis key dropped (recomputed from config)
          - target_pos_embed loaded as-is
        """
        # remove_duplicate=False ensures tied weights (e.g. lm_head.weight when
        # ParallelLMHead ties to tok_embeddings) still appear in the dict.
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_keys: set[str] = set()

        # Buffer w1 tensors until the matching w3 appears
        w1_buf: dict[str, torch.Tensor] = {}

        for name, loaded_weight in weights:
            # Drop recomputed buffer
            if name == "freqs_cis":
                continue

            # FFN w1 — buffer until we see w3
            m = re.match(r"(layers\.\d+\.feed_forward)\.w1\.weight$", name)
            if m:
                w1_buf[m.group(1)] = loaded_weight
                continue

            # FFN w3 — merge with buffered w1 → w_merged
            m = re.match(r"(layers\.\d+\.feed_forward)\.w3\.weight$", name)
            if m:
                base = m.group(1)
                merged_key = f"{base}.w_merged.weight"
                if merged_key in params_dict and base in w1_buf:
                    w1 = w1_buf.pop(base)
                    merged = torch.cat([w1, loaded_weight], dim=0)
                    param = params_dict[merged_key]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, merged.to(dtype=param.dtype))
                    loaded_keys.add(merged_key)
                continue

            # Remap checkpoint's "output.weight" → our "lm_head.weight"
            if name == "output.weight":
                name = "lm_head.weight"

            # All other weights: use weight_loader if available (e.g. ParallelLMHead
            # shards the vocab dimension); fall back to default_weight_loader.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight.to(dtype=param.dtype))
                loaded_keys.add(name)

        # Move the precomputed freqs_cis to the correct device now that
        # the engine is fully set up and we know the GPU device.
        try:
            device = next(
                p.device for p in params_dict.values()
                if p.device.type not in ("", "cpu", "meta")
            )
        except StopIteration:
            device = torch.device("cuda", 0)
        self.freqs_cis = self._freqs_cis_cpu.to(device=device)

        return loaded_keys


# ---------------------------------------------------------------------------
# Factory functions mirroring gpt_model.py  (use hf_config fields instead)
# ---------------------------------------------------------------------------

def _make_po_config(
    n_layer, n_head, dim,
    vocab_size=16384, block_size=576, num_classes=1000,
    cls_token_num=1, cfg_scale=4.0, **kwargs,
) -> ModelArgs:
    return ModelArgs(
        n_layer=n_layer, n_head=n_head, dim=dim,
        vocab_size=vocab_size, block_size=block_size,
        num_classes=num_classes, cls_token_num=cls_token_num,
        cfg_scale=cfg_scale, **kwargs,
    )


GPT_models_PO = {
    "GPT-B":    lambda **kw: _make_po_config(12, 12, 768, **kw),
    "GPT-L":    lambda **kw: _make_po_config(24, 16, 1024, **kw),
    "GPT-XL":   lambda **kw: _make_po_config(36, 20, 1280, **kw),
    "GPT-XXL":  lambda **kw: _make_po_config(48, 24, 1536, **kw),
    "GPT-XXXL": lambda **kw: _make_po_config(48, 40, 2560, **kw),
    "GPT-1B":   lambda **kw: _make_po_config(22, 32, 2048, **kw),
    "GPT-3B":   lambda **kw: _make_po_config(24, 32, 3200, **kw),
    "GPT-7B":   lambda **kw: _make_po_config(32, 32, 4096, **kw),
}
