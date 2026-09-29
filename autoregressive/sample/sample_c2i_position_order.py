# Sampling script for Position Order GPT
# Modified from sample_c2i.py to support token_order-aware generation
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
from autoregressive.models.generate_position_order import generate


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        position_order=args.position_order,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:  # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        # Assume raw state dict (flat parameter names as keys)
        model_weight = checkpoint
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model, mode="reduce-overhead", fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo")

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

    # Build token_order
    block_size = latent_size**2
    bsz = len(class_labels)

    if args.token_order_file:
        # Load from file
        if args.token_order_file.endswith(".npy"):
            token_order = torch.from_numpy(np.load(args.token_order_file)).long().to(device)
        else:
            token_order = torch.load(args.token_order_file, map_location=device).long()
        # Expand to batch if 1D
        if token_order.dim() == 1:
            token_order = token_order.unsqueeze(0).expand(bsz, -1).contiguous()
        assert token_order.shape == (bsz, block_size), \
            f"token_order shape {token_order.shape} != expected ({bsz}, {block_size})"
        print(f"loaded token_order from {args.token_order_file}")
    elif args.position_order == "raster":
        token_order = torch.arange(block_size, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        print(f"using raster token order")
    else:
        # Random order
        if args.order_seed is not None:
            rng = torch.Generator(device='cpu')
            rng.manual_seed(args.order_seed)
            token_order = torch.stack([
                torch.randperm(block_size, generator=rng)
                for _ in range(bsz)
            ]).to(device)
            print(f"using random token order with seed {args.order_seed}")
        else:
            token_order = torch.stack([
                torch.randperm(block_size, device=device)
                for _ in range(bsz)
            ])
            print(f"using random token order (no seed)")

    t1 = time.time()
    index_sample = generate(
        gpt_model,
        c_indices,
        latent_size**2,
        token_order=token_order,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=True,
    )
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")

    t2 = time.time()
    samples = vq_model.decode_code(
        index_sample, qzshape
    )  # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_image(
        samples,
        "sample_{}_position_order.png".format(args.gpt_type),
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"image is saved to sample_{args.gpt_type}_position_order.png")

    # Save combined grid: generated images + token order heatmaps
    nrow = 4
    nsamples = samples.shape[0]
    ncol = nrow
    nrow_grid = int(np.ceil(nsamples / ncol))

    fig = plt.figure(figsize=(ncol * 6, nrow_grid * 3))
    gs = gridspec.GridSpec(nrow_grid, ncol * 2, wspace=0.05, hspace=0.3)

    # Convert generated images to numpy for plotting
    samples_np = torch.clamp(samples * 0.5 + 0.5, 0, 1).cpu().permute(0, 2, 3, 1).numpy()
    token_order_cpu = token_order.cpu()

    for i in range(nsamples):
        row = i // ncol
        col = i % ncol

        # Generated image
        ax_img = fig.add_subplot(gs[row, col * 2])
        ax_img.imshow(samples_np[i])
        ax_img.set_title(f"class {class_labels[i]}", fontsize=9)
        ax_img.axis("off")

        # Token order heatmap: token_order[i] is [block_size] where
        # token_order[i][step] = spatial_position generated at that step.
        # We want to show for each spatial position, when it was generated.
        inv_order = torch.zeros(block_size, dtype=torch.long)
        inv_order.scatter_(0, token_order_cpu[i], torch.arange(block_size))
        order_map = inv_order.reshape(latent_size, latent_size).float().numpy()

        ax_order = fig.add_subplot(gs[row, col * 2 + 1])
        ax_order.imshow(order_map, cmap="viridis", interpolation="nearest")
        ax_order.set_title("gen order", fontsize=9)
        ax_order.axis("off")

    order_path = f"sample_{args.gpt_type}_position_order_vis.png"
    fig.savefig(order_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"order visualization saved to {order_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L"
    )
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument(
        "--gpt-type",
        type=str,
        choices=["c2i", "t2i"],
        default="c2i",
        help="class-conditional or text-conditional",
    )
    parser.add_argument("--from-fsdp", action="store_true")
    parser.add_argument(
        "--cls-token-num",
        type=int,
        default=1,
        help="max token number of condition input",
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"]
    )
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16"
    )
    parser.add_argument(
        "--vq-ckpt", type=str, default=None, help="ckpt path for vq model"
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=16384,
        help="codebook size for vector quantization",
    )
    parser.add_argument(
        "--codebook-embed-dim",
        type=int,
        default=8,
        help="codebook dimension for vector quantization",
    )
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--top-k", type=int, default=2000, help="top-k value to sample with"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature value to sample with",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="top-p value to sample with"
    )
    # Position order arguments
    parser.add_argument(
        "--position-order",
        type=str,
        choices=["random", "raster"],
        default="random",
        help="token ordering strategy: random or raster",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=None,
        help="seed for deterministic random token order (optional)",
    )
    parser.add_argument(
        "--token-order-file",
        type=str,
        default=None,
        help="path to .pt or .npy file with precomputed token order",
    )
    args = parser.parse_args()
    main(args)
