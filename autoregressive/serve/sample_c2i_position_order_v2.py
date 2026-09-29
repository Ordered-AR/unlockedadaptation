# Sampling script for position-order vLLM 0.13.0 serving.
#
# Key differences from sample_c2i_position_order.py (vLLM 0.4.x):
#   - Uses vLLM 0.13.0 LLM class directly (no custom LLM subclass needed)
#   - Registers TransformerPositionOrder with vLLM's ModelRegistry
#   - Injects token_order into the model via LLM.apply_model() before generate()
#   - Applies inverse permutation after generation to restore raster order
#
# vLLM 0.13.0 v1 engine runs the model in a subprocess; apply_model() must
# serialise the function over IPC.  VLLM_ALLOW_INSECURE_SERIALIZATION=1
# switches to pickle (which handles closures) instead of the default encoder.
import os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import json
import shutil
import tempfile
import time
import argparse

import torch
from torchvision.utils import save_image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.serve.gpt_model_position_order_v2 import (
    GPT_models_PO,
    TransformerPositionOrder,
)

# Register our custom model with vLLM before importing LLM
from vllm.model_executor.models.registry import ModelRegistry
ModelRegistry.register_model("TransformerPositionOrder", TransformerPositionOrder)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


# ---------------------------------------------------------------------------
# Token order utilities
# ---------------------------------------------------------------------------

def generate_token_order(
    batch_size: int,
    block_size: int,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Return [batch_size, block_size] CPU long tensor of spatial generation orders.

    mode='raster'  → all rows are [0, 1, …, block_size-1]
    mode='random'  → each row is an independent randperm (seeded by seed)
    """
    if mode == "raster":
        return (
            torch.arange(block_size, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.stack(
        [torch.randperm(block_size, generator=rng) for _ in range(batch_size)]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load VQ tokenizer
    # ------------------------------------------------------------------
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print("image tokenizer is loaded")

    # ------------------------------------------------------------------
    # Sampling setup
    # ------------------------------------------------------------------
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2
    bsz = len(class_labels)
    qzshape = [bsz, args.codebook_embed_dim, latent_size, latent_size]

    # Generate per-sample token_orders
    token_order = generate_token_order(bsz, block_size, args.position_order, args.order_seed)
    # [N, block_size], CPU

    # Build prompt token IDs
    prompt_token_ids = [[cind] for cind in class_labels]

    # CFG: double the batch — unconditioned sequences use the null class index
    if args.cfg_scale > 1.0:
        prompt_token_ids.extend([[args.num_classes] for _ in range(bsz)])
        token_orders_full = torch.cat([token_order, token_order], dim=0)  # [2N, block_size]
    else:
        token_orders_full = token_order  # [N, block_size]

    # ------------------------------------------------------------------
    # Build a temporary config directory with the correct block_size.
    # block_size is derived from image_size / downsample_size so it matches
    # the checkpoint (the base fake_json dirs hardcode one image resolution).
    # ------------------------------------------------------------------
    base_cfg_path = "autoregressive/serve/fake_json/{}-PO/config.json".format(args.gpt_model)
    with open(base_cfg_path) as f:
        cfg = json.load(f)
    cfg["block_size"] = block_size  # latent_size ** 2, computed above
    cfg["cls_token_num"] = args.cls_token_num
    cfg["num_classes"] = args.num_classes
    cfg["cfg_scale"] = args.cfg_scale

    tmp_model_dir = tempfile.mkdtemp(prefix="vllm_po_cfg_")
    with open(os.path.join(tmp_model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    try:
        # ------------------------------------------------------------------
        # Load LLM (vLLM 0.13.0)
        #
        # load_format="dummy": vLLM initialises parameters to random values
        # and skips all weight file loading.  We then load the real checkpoint
        # via apply_model() after the engine is fully set up.
        # ------------------------------------------------------------------
        llm = LLM(
            model=tmp_model_dir,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.9,
            enforce_eager=True,           # required: dynamic token_order breaks CUDA graph
            dtype="bfloat16",
            trust_remote_code=False,
            load_format="dummy",
            enable_prefix_caching=False,  # no benefit for unique image tokens; saves overhead
            enable_chunked_prefill=False, # simpler scheduling for homogeneous batch
        )
        print("gpt model structure initialised (random weights)")

        # Load the real checkpoint weights into the model.
        def _load_checkpoint(model):
            checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
            if "model" in checkpoint:
                model_weight = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model_weight = checkpoint["state_dict"]
            else:
                # Raw state dict (OrderedDict of tensors) or --from-fsdp flat dict
                model_weight = checkpoint
            model.load_weights(model_weight.items())
            del checkpoint

        llm.apply_model(_load_checkpoint)
        print("checkpoint loaded")

        # Inject token_order into the model.
        # All sequences are at the same decode step at every engine step (vLLM
        # guarantee for a homogeneous batch), so a single [2N, block_size] tensor
        # covers the whole generation.
        #
        # IMPORTANT: the model runs in a worker subprocess (spawn).  We must pass
        # a CPU tensor in the closure and move it to CUDA *inside* the worker so
        # that the CUDA context is the worker's own context.  Passing a CUDA tensor
        # directly causes silent corruption via IPC.
        token_orders_cpu = token_orders_full.cpu()
        llm.apply_model(lambda model: setattr(model, "token_order", token_orders_cpu.cuda()))
        # cond_bsz tells compute_logits how to split the batch for CFG combination
        _bsz = bsz
        llm.apply_model(lambda model: setattr(model, "cond_bsz", _bsz))
        print("token_order injected")

        # ------------------------------------------------------------------
        # Sampling params
        # ------------------------------------------------------------------
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=block_size,
            # ignore_eos=True: the fake config sets eos_token_id=2, but token 2 is a
            # valid VQ codebook token.  Without this, vLLM would stop generation early
            # whenever it samples token 2.
            ignore_eos=True,
        )

        # ------------------------------------------------------------------
        # Generate
        # ------------------------------------------------------------------
        # Set seed before generate to make vLLM's internal RNG deterministic
        torch.manual_seed(args.seed)
        t1 = time.time()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        sampling_time = time.time() - t1
        print(f"gpt sampling takes about {sampling_time:.2f} seconds.")

        # ------------------------------------------------------------------
        # Collect generated sequences (step-order)
        # ------------------------------------------------------------------
        generated = torch.tensor(
            [output.outputs[0].token_ids for output in outputs], device=device
        )  # [2N or N, block_size] — tokens in generation step order

        # Keep only the conditioned half (first N rows)
        if args.cfg_scale > 1.0:
            generated = generated[:bsz]  # [N, block_size]

        # ------------------------------------------------------------------
        # Inverse permutation: step-order → raster order
        #
        # token_order[b, step] = raster_pos
        #   → index_sample[b, raster_pos] = generated[b, step]
        # Implemented as scatter along dim=1.
        # ------------------------------------------------------------------
        to_permute = token_order.to(device)  # [N, block_size]
        index_sample = torch.zeros_like(generated)
        index_sample.scatter_(1, to_permute, generated)

        # ------------------------------------------------------------------
        # Decode to images
        # ------------------------------------------------------------------
        t2 = time.time()
        samples = vq_model.decode_code(index_sample, qzshape)
        decoder_time = time.time() - t2
        print(f"decoder takes about {decoder_time:.2f} seconds.")

        save_image(
            samples,
            f"sample_{args.gpt_type}_position_order_vllm_v2.png",
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"image saved to sample_{args.gpt_type}_position_order_vllm_v2.png")

    finally:
        shutil.rmtree(tmp_model_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt-model", type=str, choices=list(GPT_models_PO.keys()), default="GPT-L"
    )
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument(
        "--gpt-type",
        type=str,
        choices=["c2i"],
        default="c2i",
    )
    parser.add_argument("--from-fsdp", action="store_true")
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"]
    )
    parser.add_argument(
        "--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16"
    )
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--position-order",
        type=str,
        choices=["random", "raster"],
        default="random",
        help="Token generation order: random (independent per sample) or raster",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=42,
        help="Seed for random token order generation",
    )
    args = parser.parse_args()
    main(args)
