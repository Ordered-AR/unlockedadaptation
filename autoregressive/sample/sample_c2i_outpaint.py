# Outpainting script for Position Order GPT
# Encodes an input image, keeps only a bounding box region, and regenerates
# all tokens outside the bbox autoregressively in random order.
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
import time
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import sys
from tokenizer.tokenizer_image.vq_model import VQ_models

# Switch between base GPT and position-order GPT based on --base-model flag
if "--base-model" in sys.argv:
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import (
        generate,
        prefill,
        decode_one_token,
        sample,
    )
else:
    from autoregressive.models.gpt_position_order import GPT_models
    from autoregressive.models.generate_position_order import (
        generate,
        prefill,
        decode_one_token,
        sample,
    )


@torch.no_grad()
def generate_masked(
    model,
    cond,
    image_tokens,
    regen_mask,
    token_order=None,
    cfg_scale=1.0,
    cfg_interval=-1,
    **sampling_kwargs,
):
    """
    Mask-conditional generation: feed known tokens (where regen_mask is False)
    to build context, then sample only the masked tokens.

    Args:
        model: PositionOrderGPT model
        cond: [bsz] condition token indices (class labels)
        image_tokens: [bsz, block_size] VQ token indices of the original image
        regen_mask: [block_size] bool tensor — True = regenerate, False = keep
        token_order: [bsz, block_size] generation order (known first, then unknown)
        cfg_scale: classifier-free guidance scale
        cfg_interval: after this many steps, disable CFG
        **sampling_kwargs: temperature, top_k, top_p, sample_logits

    Returns:
        result_tokens: [bsz, block_size] tokens in raster order (known kept, masked regenerated)
    """
    block_size = image_tokens.shape[1]
    bsz = cond.shape[0]
    device = cond.device

    if token_order is None:
        known_positions = (~regen_mask).nonzero(as_tuple=False).squeeze(-1)
        unknown_positions = regen_mask.nonzero(as_tuple=False).squeeze(-1)
        unknown_positions = unknown_positions[torch.randperm(len(unknown_positions), device=device)]
        order_1d = torch.cat([known_positions, unknown_positions])
        token_order = order_1d.unsqueeze(0).expand(bsz, -1).contiguous()

    num_known = (~regen_mask).sum().item()
    num_unknown = regen_mask.sum().item()

    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    T = 1
    T_new = T + block_size
    max_batch_size_cfg = bsz * 2 if cfg_scale > 1.0 else bsz

    with torch.device(device):
        model.setup_caches(
            max_batch_size=max_batch_size_cfg,
            max_seq_length=T_new,
            dtype=model.tok_embeddings.weight.dtype,
        )

    input_pos = torch.arange(0, T, device=device)
    prefill_kwargs = sampling_kwargs.copy()
    if hasattr(model, 'position_order'):
        prefill_kwargs["token_order"] = token_order
    first_logits_token = prefill(
        model, cond_combined, input_pos, cfg_scale,
        **prefill_kwargs,
    )

    generated_seq = torch.empty((bsz, block_size), dtype=torch.int, device=device)

    spatial_pos_0 = token_order[:, 0]
    if not regen_mask[spatial_pos_0[0]]:
        cur_token = image_tokens.gather(1, spatial_pos_0.unsqueeze(1))
    else:
        cur_token = first_logits_token
    generated_seq[:, 0] = cur_token.squeeze(1)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    cfg_flag = True

    for step in range(1, block_size):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            if cfg_interval > -1 and step > cfg_interval:
                cfg_flag = False

            decode_kwargs = sampling_kwargs.copy()
            if hasattr(model, 'position_order'):
                decode_kwargs["token_order"] = token_order
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag,
                **decode_kwargs,
            )
            input_pos += 1

            spatial_pos = token_order[:, step]
            if step < num_known:
                forced_token = image_tokens.gather(1, spatial_pos.unsqueeze(1))
                generated_seq[:, step] = forced_token.squeeze(1)
                cur_token = forced_token
            else:
                generated_seq[:, step] = next_token.squeeze(1)
                cur_token = next_token.view(-1, 1)

    inv_order = torch.zeros_like(token_order)
    inv_order.scatter_(
        1, token_order,
        torch.arange(block_size, device=device).unsqueeze(0).expand_as(token_order),
    )
    raster_seq = torch.gather(generated_seq, 1, inv_order)

    return raster_seq


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size ** 2
    model_kwargs = dict(
        vocab_size=args.codebook_size,
        block_size=block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    )
    if not args.base_model:
        model_kwargs["position_order"] = args.position_order
    gpt_model = GPT_models[args.gpt_model](**model_kwargs).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:
        model_weight = checkpoint
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        model_weight = checkpoint
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    if args.image_path:
        image_size = args.image_size
        img = Image.open(args.image_path).convert("RGB").resize((image_size, image_size))
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        h = vq_model.encoder(img_tensor)
        h = vq_model.quant_conv(h)
        _, _, (_, _, token_indices) = vq_model.quantize(h)
        image_tokens = token_indices.reshape(1, -1).to(device)
        print(f"encoded input image to {image_tokens.shape[1]} tokens")
    else:
        print("no --image-path provided, generating image from scratch...")
        c_gen = torch.tensor([args.class_label], device=device)
        gen_kwargs = dict(
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        if not args.base_model:
            gen_order = torch.stack([torch.randperm(block_size, device=device)])
            gen_kwargs["token_order"] = gen_order
        image_tokens = generate(
            gpt_model,
            c_gen,
            block_size,
            **gen_kwargs,
        )
        print(f"generated image with {block_size} tokens")

    # Parse bounding box in patch coordinates — bbox is the KEPT region
    bbox = [int(x) for x in args.bbox.split(",")]
    assert len(bbox) == 4, "bbox must be lx,ly,rx,ry in patch coords"
    lx, ly, rx, ry = bbox

    lx = max(0, min(lx, latent_size))
    ly = max(0, min(ly, latent_size))
    rx = max(0, min(rx, latent_size))
    ry = max(0, min(ry, latent_size))

    keep_mask = torch.zeros(latent_size, latent_size, dtype=torch.bool, device=device)
    keep_mask[ly:ry, lx:rx] = True
    keep_mask_flat = keep_mask.flatten()
    regen_mask_flat = ~keep_mask_flat  # True = regenerate (outside bbox)

    num_kept = keep_mask_flat.sum().item()
    num_regen = regen_mask_flat.sum().item()
    if num_kept == 0 or num_regen == 0:
        raise ValueError(
            f"outpainting requires a non-empty kept region and non-empty regen region "
            f"(got kept={num_kept}, regen={num_regen}); bbox was ({lx},{ly})-({rx},{ry}) "
            f"on a {latent_size}x{latent_size} grid"
        )
    print(f"bbox patch ({lx},{ly})-({rx},{ry}) on {latent_size}x{latent_size} grid")
    print(f"keeping {num_kept}/{block_size} tokens, regenerating {num_regen} (outpainting)")

    num_samples = args.num_samples
    c_indices = torch.tensor([args.class_label] * num_samples, device=device)
    image_tokens_batch = image_tokens.expand(num_samples, -1).contiguous()
    qzshape = [num_samples, args.codebook_embed_dim, latent_size, latent_size]

    if args.order_seed is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(args.order_seed)

    known_positions = (~regen_mask_flat).nonzero(as_tuple=False).squeeze(-1)
    unknown_positions = regen_mask_flat.nonzero(as_tuple=False).squeeze(-1)

    orders = []
    for i in range(num_samples):
        if args.order_seed is not None:
            perm = torch.randperm(len(unknown_positions), generator=rng)
        else:
            perm = torch.randperm(len(unknown_positions), device=device)
        shuffled_unknown = unknown_positions[perm.to(device)]
        order_1d = torch.cat([known_positions, shuffled_unknown])
        orders.append(order_1d)
    token_order = torch.stack(orders).to(device)

    t1 = time.time()
    result_tokens = generate_masked(
        gpt_model,
        c_indices,
        image_tokens_batch,
        regen_mask_flat,
        token_order=token_order,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=True,
    )
    sampling_time = time.time() - t1
    print(f"outpainting takes about {sampling_time:.2f} seconds.")

    for i in range(num_samples):
        orig = image_tokens_batch[i, keep_mask_flat]
        res = result_tokens[i, keep_mask_flat]
        match = (orig == res).all().item()
        print(f"  sample {i}: kept tokens preserved = {match}"
              f" ({(orig == res).sum().item()}/{keep_mask_flat.sum().item()})")

    t2 = time.time()
    result_images = vq_model.decode_code(result_tokens, qzshape)
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    orig_qzshape = [1, args.codebook_embed_dim, latent_size, latent_size]
    orig_decoded = vq_model.decode_code(image_tokens, orig_qzshape)

    save_image(
        result_images,
        f"outpaint_{args.gpt_type}_result.png",
        nrow=num_samples,
        normalize=True,
        value_range=(-1, 1),
    )

    ncol = 4
    nrow_grid = num_samples

    fig = plt.figure(figsize=(ncol * 4, nrow_grid * 4))
    gs = gridspec.GridSpec(nrow_grid, ncol, wspace=0.08, hspace=0.3)

    orig_np = torch.clamp(orig_decoded[0] * 0.5 + 0.5, 0, 1).cpu().permute(1, 2, 0).numpy()
    result_np = torch.clamp(result_images * 0.5 + 0.5, 0, 1).cpu().permute(0, 2, 3, 1).numpy()

    ds = args.downsample_size
    px_lx, px_ly = lx * ds, ly * ds
    px_rx, px_ry = rx * ds, ry * ds
    img_px = args.image_size

    for i in range(num_samples):
        ax_orig = fig.add_subplot(gs[i, 0])
        ax_orig.imshow(orig_np)
        ax_orig.set_title("Original", fontsize=11)
        ax_orig.axis("off")

        # Input column: show only the kept bbox content on a black background
        ax_masked = fig.add_subplot(gs[i, 1])
        input_view = np.zeros_like(orig_np)
        input_view[px_ly:px_ry, px_lx:px_rx] = orig_np[px_ly:px_ry, px_lx:px_rx]
        ax_masked.imshow(input_view)
        ax_masked.set_title("Input", fontsize=11)
        ax_masked.axis("off")

        ax_order = fig.add_subplot(gs[i, 2])
        inv_order = torch.zeros(block_size, dtype=torch.long)
        token_order_cpu = token_order[i].cpu()
        inv_order.scatter_(0, token_order_cpu, torch.arange(block_size))
        order_map = inv_order.reshape(latent_size, latent_size).float().numpy()

        regen_mask_np = regen_mask_flat.reshape(latent_size, latent_size).cpu().numpy()
        display_map = np.full((latent_size, latent_size, 4), 0.7)
        unknown_order_vals = order_map[regen_mask_np]
        if len(unknown_order_vals) > 0:
            vmin = unknown_order_vals.min()
            vmax = unknown_order_vals.max()
            cmap = plt.cm.viridis
            for r in range(latent_size):
                for c in range(latent_size):
                    if regen_mask_np[r, c]:
                        normed = (order_map[r, c] - vmin) / max(vmax - vmin, 1)
                        display_map[r, c] = cmap(normed)

        ax_order.imshow(display_map, interpolation="nearest")
        ax_order.set_title("Order", fontsize=11)
        ax_order.axis("off")

        ax_result = fig.add_subplot(gs[i, 3])
        ax_result.imshow(result_np[i])
        ax_result.set_title(f"class {args.class_label}", fontsize=11)
        ax_result.axis("off")

    vis_path = f"outpaint_{args.gpt_type}_vis.png"
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"visualization saved to {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")
    parser.add_argument("--from-fsdp", action="store_true")
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--image-path", type=str, default=None, help="path to input image (if omitted, generates one first)")
    parser.add_argument("--bbox", type=str, required=True, help="KEPT bounding box as lx,ly,rx,ry in patch coords (0 to latent_size); everything outside is regenerated")
    parser.add_argument("--class-label", type=int, required=True, help="ImageNet class index for conditioning")
    parser.add_argument("--num-samples", type=int, default=4, help="number of outpainting samples to generate")
    parser.add_argument("--position-order", type=str, choices=["random", "raster"], default="random")
    parser.add_argument("--order-seed", type=int, default=None)
    parser.add_argument("--base-model", action="store_true", help="use base gpt.py instead of gpt_position_order.py")

    args = parser.parse_args()
    main(args)
