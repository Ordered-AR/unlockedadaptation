# Training script for Position Order GPT Model
# Modified from train_c2i.py to support position-based token reordering
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import math
import wandb
from wandb import AlertLevel

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.lr_scheduler import get_scheduler
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.config_integration import apply_config_to_args
from dataset.build import build_dataset
from autoregressive.models.gpt_position_order import GPT_models
from utils.broadcast_utils import broadcast_value


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # Separate target_pos_embed into its own group with 10x LR and weight decay,
    # since it trains from scratch while the rest of the model is pretrained.
    target_pos_embed_params = [p for n, p in param_dict.items() if "target_pos_embed" in n]
    other_param_dict = {n: p for n, p in param_dict.items() if "target_pos_embed" not in n}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in other_param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in other_param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": learning_rate},
        # target_pos_embed: 10x LR to learn quickly from scratch, with weight decay for stability
        {"params": target_pos_embed_params, "weight_decay": weight_decay, "lr": learning_rate * 10},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    num_target_params = sum(p.numel() for p in target_pos_embed_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    logger.info(
        f"target_pos_embed parameters: {num_target_params:,} (lr={learning_rate * 10:.2e}, wd={weight_decay})"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(
            args.results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace(
            "/", "-"
        )  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        checkpoint_dir = None  # Initialize for other ranks
        cloud_checkpoint_dir = None
        logger = create_logger(None)

    # Broadcast checkpoint_dir and cloud_checkpoint_dir to all ranks
    checkpoint_dir = broadcast_value(checkpoint_dir, rank)
    cloud_checkpoint_dir = broadcast_value(cloud_checkpoint_dir, rank)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}."
    )

    # Initialize wandb (only on rank 0)
    if rank == 0 and args.use_wandb:
        run_name = (
            args.wandb_run_name
            if args.wandb_run_name
            else f"{model_string_name}-{args.position_order}-e{args.epochs}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            dir=experiment_dir,
        )
        logger.info(
            f"Weights & Biases initialized: project={args.wandb_project}, run={run_name}"
        )
        wandb_run.alert(
            title=f"Run started: {run_name}",
            text=(
                f"Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}\n"
                f"Node: {os.environ.get('SLURM_NODELIST', 'N/A')}\n"
                f"Model: {args.gpt_model}, Position order: {args.position_order}\n"
                f"Epochs: {args.epochs}, Batch size: {args.global_batch_size}, LR: {args.lr}"
            ),
            level=AlertLevel.INFO,
        )

    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        position_order=args.position_order,  # New parameter for position ordering
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Position order mode: {args.position_order}")
    if args.position_order == "scheduled":
        if args.position_order_transition_epochs > args.epochs:
            logger.warning(
                f"Transition epochs ({args.position_order_transition_epochs}) > total epochs ({args.epochs}). "
                f"Will use raster throughout training."
            )
        logger.info(
            f"Position order scheduling: Raster → Random over {args.position_order_transition_epochs} epochs"
        )
        init_raster = 1.0 - args.initial_random_prob
        logger.info(
            f"  - Epoch 0: {init_raster:.0%} raster / {args.initial_random_prob:.0%} random"
        )
        logger.info(
            f"  - Epochs 0-{args.position_order_transition_epochs-1}: Linear transition to 100% random"
        )
        logger.info(f"  - Epochs {args.position_order_transition_epochs}+: 100% random")

    if args.ema:
        ema = deepcopy(model).to(
            device
        )  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    if args.freeze_qkv:
        frozen = 0
        for name, param in model.named_parameters():
            if "wqkv" in name:
                param.requires_grad = False
                frozen += param.numel()
        logger.info(f"Froze wqkv projections: {frozen:,} parameters")

    # Setup optimizer
    optimizer = creat_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger
    )

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    flip_info = "with" if hasattr(dataset, "flip") and dataset.flip else "without"
    feature_dir = getattr(dataset, "feature_dir", "")
    aug_info = 10 if "ten_crop" in feature_dir else 1
    aug_feature_dir = getattr(dataset, "aug_feature_dir", None)
    aug_info = 2 * aug_info if aug_feature_dir is not None else aug_info
    logger.info(
        f"Dataset contains {len(dataset):,} images ({args.code_path}) "
        f"{flip_info} flip augmentation and {aug_info} crop augmentation"
    )

    # Create learning rate scheduler
    total_training_steps = args.epochs * math.ceil(
        len(dataset) / args.global_batch_size
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=args.lr_min_ratio,
        lr_end=args.lr_end,
        power=args.lr_power,
    )
    logger.info(
        f"Created {args.lr_scheduler} scheduler with {args.lr_warmup_steps} warmup steps and {total_training_steps} total steps"
    )

    # Prepare models for training - handle both old and new checkpoint formats
    train_steps = 0
    start_epoch = 0

    if args.resume_dir:
        # New checkpoint format: load from resume directory
        logger.info(f"Resuming from checkpoint directory: {args.resume_dir}")
        metadata = load_checkpoint(
            resume_dir=args.resume_dir,
            model=model,
            optimizer=optimizer if not args.reset_optimizer else None,
            scheduler=scheduler,
            ema=ema if args.ema else None,
            load_optimizer=not args.reset_optimizer,
            load_scheduler=True,
            load_ema=args.ema,
            logger=logger,
            device="cpu",
        )

        if "train_steps" in metadata:
            train_steps = metadata["train_steps"]
            start_epoch = int(
                train_steps / math.ceil(len(dataset) / args.global_batch_size)
            )
            logger.info(f"Resuming from step {train_steps}, epoch {start_epoch}")

    elif args.gpt_ckpt:
        # Old checkpoint format: single .pt file
        logger.info(f"Loading old-format checkpoint: {args.gpt_ckpt}")
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)

        # Load model weights (strict=False to allow new params like target_pos_embed)
        if "model" in checkpoint:
            incompatible = model.load_state_dict(checkpoint["model"], strict=False)
            logger.info(f"Loaded model weights from checkpoint")
        else:
            # Checkpoint might be just the state dict
            incompatible = model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded model state dict from checkpoint")
        if incompatible.missing_keys:
            logger.info(f"Missing keys (new params, will train from init): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")

        # Load EMA if available and requested
        if args.ema:
            if "ema" in checkpoint:
                ema.load_state_dict(checkpoint["ema"])
                logger.info(f"Loaded EMA weights from checkpoint")
            else:
                # Initialize EMA from model weights
                ema.load_state_dict(model.state_dict())
                logger.info(
                    f"Initialized EMA from model weights (no EMA in checkpoint)"
                )

        # Load optimizer state if available (optional for fine-tuning)
        if "optimizer" in checkpoint and not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded optimizer state from checkpoint")
        else:
            if args.reset_optimizer:
                logger.info(f"Skipping optimizer state (--reset-optimizer flag set)")
            else:
                logger.warning(
                    f"No optimizer state in checkpoint - starting with fresh optimizer"
                )

        # Load training progress if available
        if "steps" in checkpoint:
            train_steps = checkpoint["steps"]
            start_epoch = int(
                train_steps / math.ceil(len(dataset) / args.global_batch_size)
            )
            logger.info(f"Resuming from step {train_steps}, epoch {start_epoch}")
        else:
            # Try to parse from filename (e.g., 0010000.pt -> step 10000)
            try:
                train_steps = int(args.gpt_ckpt.split("/")[-1].split(".")[0])
                start_epoch = int(
                    train_steps / math.ceil(len(dataset) / args.global_batch_size)
                )
                logger.info(f"Inferred step {train_steps} from filename")
            except (ValueError, IndexError):
                logger.warning(f"Could not determine training step - starting from 0")

        del checkpoint
        logger.info(f"Checkpoint loaded from: {args.gpt_ckpt}")

    logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")

    # Initialize EMA if starting from scratch
    if train_steps == 0 and start_epoch == 0 and args.ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = DDP(model, device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.mixed_precision
    ]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16"))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_grad_norm = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        if (
            args.position_order == "scheduled"
            and args.position_order_transition_epochs > 0
        ):
            if epoch < args.position_order_transition_epochs:
                # Linear interpolation: random_prob goes from initial_random_prob to 1.0
                random_prob = args.initial_random_prob + (1.0 - args.initial_random_prob) * (epoch / args.position_order_transition_epochs)
                raster_prob = 1.0 - random_prob
            else:
                raster_prob = 0.0
        else:
            raster_prob = 1.0 if args.position_order == "raster" else 0.0
        logger.info(f"Beginning epoch {epoch} (raster_prob={raster_prob:.3f})...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            with torch.amp.autocast("cuda", dtype=ptdtype):
                # Note: token_order is generated internally by the model based on position_order parameter
                # The model returns (logits, loss, token_order)
                # Pass raster_prob for scheduling
                _, loss, _ = model(
                    cond_idx=c_indices,
                    idx=z_indices[:, :-1],
                    targets=z_indices,
                    raster_prob=raster_prob,
                )

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm if args.max_grad_norm != 0.0 else float("inf"),
            ).item()

            # Step the optimizer and scaler
            scaler.step(optimizer)
            scaler.update()
            # Step the learning rate scheduler
            scheduler.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            if args.ema:
                update_ema(
                    ema, model.module._orig_mod if not args.no_compile else model.module
                )

            # Log loss values:
            running_loss += loss.item()
            running_grad_norm += grad_norm
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss and grad norm over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_grad_norm = torch.tensor(
                    running_grad_norm / log_steps, device=device
                )
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_grad_norm = avg_grad_norm.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}, epoch={epoch}) Train Loss: {avg_loss:.4f}, "
                    f"Grad Norm: {avg_grad_norm:.4f}, Raster Prob: {raster_prob:.3f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank == 0 and args.use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/grad_norm": avg_grad_norm,
                            "train/epoch": epoch,
                            "train/raster_prob": raster_prob,
                            "train/steps_per_sec": steps_per_sec,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/learning_rate_target_pos_embed": optimizer.param_groups[2]["lr"],
                        },
                        step=train_steps,
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_grad_norm = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                # Save to local directory (new format)
                if not args.no_local_save:
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        train_steps=train_steps,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema if args.ema else None,
                        args=args,
                        rank=rank,
                        logger=logger,
                        keep_last_k=args.keep_last_k,
                    )

                # Save to cloud directory (new format)
                save_checkpoint(
                    checkpoint_dir=cloud_checkpoint_dir,
                    train_steps=train_steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema if args.ema else None,
                    args=args,
                    rank=rank,
                    logger=logger,
                    keep_last_k=args.keep_last_k,
                )
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and args.use_wandb:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file argument (MUST be first for proper override)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )

    # Data paths
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument(
        "--cloud-save-path",
        type=str,
        required=True,
        help="please specify a cloud disk path, if not, local path",
    )
    parser.add_argument(
        "--no-local-save",
        action="store_true",
        help="no save checkpoints to local path for limited disk volume",
    )
    parser.add_argument(
        "--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B"
    )
    parser.add_argument(
        "--gpt-ckpt", type=str, default=None, help="ckpt path for resume training"
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="reset optimizer state when loading checkpoint (useful for fine-tuning)",
    )
    parser.add_argument(
        "--gpt-type",
        type=str,
        choices=["c2i", "t2i"],
        default="c2i",
        help="class-conditional or text-conditional",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16384,
        help="vocabulary size of visual tokenizer",
    )
    parser.add_argument("--ema", action="store_true", help="whether using ema training")
    parser.add_argument(
        "--freeze-qkv",
        action="store_true",
        default=False,
        help="Freeze wqkv projections in all attention blocks during finetuning",
    )
    parser.add_argument(
        "--cls-token-num",
        type=int,
        default=1,
        help="max token number of condition input",
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.1,
        help="dropout_p of resid_dropout_p and ffn_dropout_p",
    )
    parser.add_argument(
        "--token-dropout-p",
        type=float,
        default=0.1,
        help="dropout_p of token_dropout_p",
    )
    parser.add_argument(
        "--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay"
    )
    parser.add_argument(
        "--position-order",
        type=str,
        choices=["random", "raster", "scheduled"],
        default="random",
        help="token position order: random for random permutation, raster for standard raster scan order, scheduled for curriculum learning",
    )
    parser.add_argument(
        "--position-order-transition-epochs",
        type=int,
        default=100,
        help="number of epochs to transition from raster to random order (only used with --position-order scheduled)",
    )
    parser.add_argument(
        "--initial-random-prob",
        type=float,
        default=0.0,
        help="initial probability of random order at epoch 0 (only used with --position-order scheduled). "
             "Curriculum linearly interpolates from initial_random_prob to 1.0 over transition_epochs.",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="imagenet_code")
    parser.add_argument(
        "--image-size", type=int, choices=[256, 384, 448, 512], default=256
    )
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight-decay", type=float, default=5e-2, help="Weight decay to use"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 parameter for the Adam optimizer",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="beta2 parameter for the Adam optimizer",
    )
    parser.add_argument(
        "--max-grad-norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=-1,
        help="keep only last k checkpoints (-1 to keep all)",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"]
    )

    # Learning rate scheduler arguments
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=[
            "constant",
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant_with_warmup",
            "exponential_with_warmup",
        ],
        help="learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="number of warmup steps for lr scheduler",
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.05,
        help="minimum lr ratio for cosine scheduler",
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=1e-7,
        help="end lr for polynomial/exponential scheduler",
    )
    parser.add_argument(
        "--lr-power",
        type=float,
        default=1.0,
        help="power factor for polynomial scheduler",
    )

    # Checkpoint resuming
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="directory to resume training from (with model.pt, optimizer.pt, scheduler.pt)",
    )

    # Wandb arguments
    parser.add_argument(
        "--use-wandb", action="store_true", help="enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="position-order-gpt",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="wandb run name (auto-generated if not specified)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Apply config file if provided (CLI args override config values)
    if args.config:
        args = apply_config_to_args(args)
        print(f"Loaded config from: {args.config}")
        print("Note: CLI arguments override config values")

    main(args)
