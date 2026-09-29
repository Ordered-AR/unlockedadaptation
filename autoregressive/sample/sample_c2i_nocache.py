# Sampling script for no-cache parallel decoding.
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt_position_order import GPT_models
from autoregressive.models.generate_nocache import generate_nocache, make_schedule


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(ckpt["model"])
    del ckpt
    print("image tokenizer loaded")

    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    block_size = latent_size**2

    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        position_order=args.position_order,
    ).to(device=device, dtype=precision)

    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:
        sd = ckpt
    elif "model" in ckpt:
        sd = ckpt["model"]
    elif "module" in ckpt:
        sd = ckpt["module"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    gpt_model.load_state_dict(sd, strict=False)
    gpt_model.eval()
    del ckpt
    print("gpt model loaded")

    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)
    bsz = len(class_labels)
    qzshape = [bsz, args.codebook_embed_dim, latent_size, latent_size]

    # token order
    if args.position_order == "raster":
        token_order = torch.arange(block_size, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        print("using raster token order")
    else:
        if args.order_seed is not None:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(args.order_seed)
            token_order = torch.stack([
                torch.randperm(block_size, generator=rng) for _ in range(bsz)
            ]).to(device)
            print(f"random token order with seed {args.order_seed}")
        else:
            token_order = torch.stack([
                torch.randperm(block_size, device=device) for _ in range(bsz)
            ])
            print("random token order (no seed)")

    sched = make_schedule(block_size, args.num_steps, schedule=args.schedule)
    print(f"num_steps={args.num_steps} schedule={args.schedule} indep={args.independent_chunk}: "
          f"{block_size} tokens in {len(sched)} actual steps, max chunk={max(sched)}, "
          f"head={sched[:8]}{'...' if len(sched) > 8 else ''}")

    t0 = time.time()
    if device == "cuda":
        torch.cuda.synchronize()
    index_sample = generate_nocache(
        gpt_model,
        c_indices,
        block_size,
        token_order=token_order,
        num_steps=args.num_steps,
        schedule=args.schedule,
        independent_chunk=args.independent_chunk,
        placeholder_id=args.placeholder_id,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"sampling: {time.time()-t0:.2f}s ({len(sched)} forward passes)")

    t1 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"decoder: {time.time()-t1:.2f}s")

    tag = f"steps{args.num_steps}_{args.schedule}_indep{int(args.independent_chunk)}"
    out_path = f"sample_{args.gpt_type}_nocache_{tag}.png"
    save_image(samples, out_path, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image saved to {out_path}")

    # gen-order viz
    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(2, 8, wspace=0.05, hspace=0.3)
    samples_np = torch.clamp(samples * 0.5 + 0.5, 0, 1).cpu().permute(0, 2, 3, 1).numpy()
    token_order_cpu = token_order.cpu()
    for i in range(bsz):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(samples_np[i])
        ax.set_title(f"cls {class_labels[i]}", fontsize=8)
        ax.axis("off")
        inv = torch.zeros(block_size, dtype=torch.long)
        inv.scatter_(0, token_order_cpu[i], torch.arange(block_size))
        order_map = inv.reshape(latent_size, latent_size).float().numpy()
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(order_map, cmap="viridis", interpolation="nearest")
        ax2.set_title("gen order", fontsize=8)
        ax2.axis("off")
    vis_path = f"sample_{args.gpt_type}_nocache_{tag}_vis.png"
    fig.savefig(vis_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"viz saved to {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")
    parser.add_argument("--from-fsdp", action="store_true")
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None)
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
    parser.add_argument("--position-order", type=str, choices=["random", "raster"], default="random")
    parser.add_argument("--order-seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=64,
                        help="total forward passes (256=fully sequential)")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "exponential"])
    parser.add_argument("--independent-chunk", action="store_true", default=True,
                        help="chunk positions see only prefix (default True). Use --causal-chunk to disable.")
    parser.add_argument("--causal-chunk", dest="independent_chunk", action="store_false",
                        help="placeholders attend to prior placeholders (fused sequential)")
    parser.add_argument("--placeholder-id", type=int, default=0,
                        help="token id used for placeholder positions in chunks")
    args = parser.parse_args()
    main(args)
