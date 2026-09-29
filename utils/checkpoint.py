"""
Checkpoint utilities for saving and loading training state.
Supports resuming training from a directory with separate files for model, optimizer, and scheduler.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
import glob
import shutil


def cleanup_old_checkpoints(
    checkpoint_dir: str, keep_last_k: int, logger: Optional[Any] = None
):
    """
    Remove old checkpoints, keeping only the last k.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories
        keep_last_k: Number of recent checkpoints to keep (-1 to keep all)
        logger: Optional logger for output
    """
    if keep_last_k <= 0:  # -1 or 0 means keep all
        return

    # Find all checkpoint directories (step_NNNNNN format)
    checkpoint_pattern = os.path.join(checkpoint_dir, "step_*")
    checkpoint_dirs = sorted(glob.glob(checkpoint_pattern))

    if len(checkpoint_dirs) <= keep_last_k:
        return  # Nothing to delete

    # Calculate how many to delete
    num_to_delete = len(checkpoint_dirs) - keep_last_k
    dirs_to_delete = checkpoint_dirs[:num_to_delete]

    for old_dir in dirs_to_delete:
        try:
            shutil.rmtree(old_dir)
            if logger:
                logger.info(f"Deleted old checkpoint: {old_dir}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to delete checkpoint {old_dir}: {e}")


def save_checkpoint(
    checkpoint_dir: str,
    train_steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    ema: Optional[torch.nn.Module] = None,
    args: Optional[Any] = None,
    rank: int = 0,
    logger: Optional[Any] = None,
    keep_last_k: int = -1,
):
    """
    Save checkpoint as separate files for model, optimizer, scheduler, and EMA.
    Optionally removes old checkpoints to save disk space.

    Args:
        checkpoint_dir: Directory to save checkpoint files
        train_steps: Current training step
        model: Model to save (will extract state_dict)
        optimizer: Optimizer to save
        scheduler: Optional learning rate scheduler
        ema: Optional EMA model
        args: Optional training arguments
        rank: Distributed rank (only rank 0 saves)
        logger: Optional logger for logging messages
        keep_last_k: Number of recent checkpoints to keep (-1 to keep all, 0 to keep all)
    """
    if rank != 0:
        return

    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    step_dir = os.path.join(checkpoint_dir, f"step_{train_steps:07d}")
    os.makedirs(step_dir, exist_ok=True)

    # Extract model state dict (handle DDP wrapper)
    if hasattr(model, "module"):
        if hasattr(model.module, "_orig_mod"):
            # Compiled model
            model_state_dict = model.module._orig_mod.state_dict()
        else:
            # Regular DDP
            model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # Save model weights
    model_path = os.path.join(step_dir, "model.pt")
    torch.save(model_state_dict, model_path)
    if logger:
        logger.info(f"Saved model to {model_path}")

    # Save optimizer state
    optimizer_path = os.path.join(step_dir, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)
    if logger:
        logger.info(f"Saved optimizer to {optimizer_path}")

    # Save scheduler state if provided
    if scheduler is not None:
        scheduler_path = os.path.join(step_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_path)
        if logger:
            logger.info(f"Saved scheduler to {scheduler_path}")

    # Save EMA if provided
    if ema is not None:
        ema_path = os.path.join(step_dir, "ema.pt")
        torch.save(ema.state_dict(), ema_path)
        if logger:
            logger.info(f"Saved EMA to {ema_path}")

    # Save metadata (training step, args, etc.)
    metadata = {
        "train_steps": train_steps,
        "args": args,
    }
    metadata_path = os.path.join(step_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    if logger:
        logger.info(f"Saved metadata to {metadata_path}")

    # Create a "latest" symlink or marker file
    latest_path = os.path.join(checkpoint_dir, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(f"step_{train_steps:07d}")

    if logger:
        logger.info(f"Checkpoint saved at step {train_steps}")

    # Clean up old checkpoints if keep_last_k is specified
    if keep_last_k > 0:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_k, logger)


def load_checkpoint(
    resume_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema: Optional[torch.nn.Module] = None,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_ema: bool = True,
    logger: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load checkpoint from a directory with separate files.

    Args:
        resume_dir: Directory containing checkpoint files (e.g., step_0010000/)
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        ema: Optional EMA model to load state into
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        load_ema: Whether to load EMA state
        logger: Optional logger for logging messages
        device: Device to map tensors to

    Returns:
        Dictionary containing metadata (train_steps, args, etc.)
    """
    if not os.path.exists(resume_dir):
        raise ValueError(f"Resume directory does not exist: {resume_dir}")

    # If resume_dir points to a checkpoint root, try to find latest checkpoint
    if os.path.isfile(os.path.join(resume_dir, "latest.txt")):
        with open(os.path.join(resume_dir, "latest.txt"), "r") as f:
            latest_step = f.read().strip()
        resume_dir = os.path.join(resume_dir, latest_step)
        if logger:
            logger.info(f"Loading latest checkpoint from {resume_dir}")

    # Load model weights
    model_path = os.path.join(resume_dir, "model.pt")
    if not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found: {model_path}")

    model_state_dict = torch.load(model_path, map_location=device)

    # Handle DDP wrapper when loading
    if hasattr(model, "module"):
        if hasattr(model.module, "_orig_mod"):
            incompatible = model.module._orig_mod.load_state_dict(model_state_dict, strict=False)
        else:
            incompatible = model.module.load_state_dict(model_state_dict, strict=False)
    else:
        incompatible = model.load_state_dict(model_state_dict, strict=False)

    if logger:
        logger.info(f"Loaded model weights from {model_path}")
        if incompatible.missing_keys:
            logger.info(f"Missing keys (new params, will train from init): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")

    # Load optimizer state
    if load_optimizer and optimizer is not None:
        optimizer_path = os.path.join(resume_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state_dict = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state_dict)
            if logger:
                logger.info(f"Loaded optimizer state from {optimizer_path}")
        else:
            if logger:
                logger.warning(f"Optimizer checkpoint not found: {optimizer_path}")

    # Load scheduler state
    if load_scheduler and scheduler is not None:
        scheduler_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler_state_dict = torch.load(scheduler_path, map_location=device)
            scheduler.load_state_dict(scheduler_state_dict)
            if logger:
                logger.info(f"Loaded scheduler state from {scheduler_path}")
        else:
            if logger:
                logger.warning(f"Scheduler checkpoint not found: {scheduler_path}")

    # Load EMA state
    if load_ema and ema is not None:
        ema_path = os.path.join(resume_dir, "ema.pt")
        if os.path.exists(ema_path):
            ema_state_dict = torch.load(ema_path, map_location=device)
            ema.load_state_dict(ema_state_dict)
            if logger:
                logger.info(f"Loaded EMA state from {ema_path}")
        else:
            if logger:
                logger.warning(f"EMA checkpoint not found: {ema_path}")

    # Load metadata
    metadata_path = os.path.join(resume_dir, "metadata.pt")
    metadata = {}
    if os.path.exists(metadata_path):
        # PyTorch 2.6+ defaults to weights_only=True, but our metadata contains argparse.Namespace
        # Since this is our own checkpoint, it's safe to use weights_only=False
        metadata = torch.load(metadata_path, map_location=device, weights_only=False)
        if logger:
            logger.info(f"Loaded metadata from {metadata_path}")
    else:
        if logger:
            logger.warning(f"Metadata not found: {metadata_path}")
        # Try to infer train_steps from directory name
        import re

        match = re.search(r"step_(\d+)", resume_dir)
        if match:
            metadata["train_steps"] = int(match.group(1))

    return metadata


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Root checkpoint directory

    Returns:
        Path to latest checkpoint directory, or None if not found
    """
    # First check for latest.txt marker
    latest_file = os.path.join(checkpoint_dir, "latest.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            latest_step = f.read().strip()
        return os.path.join(checkpoint_dir, latest_step)

    # Otherwise, find all step_* directories
    step_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if not step_dirs:
        return None

    # Sort by step number
    import re

    def extract_step(path):
        match = re.search(r"step_(\d+)", path)
        return int(match.group(1)) if match else 0

    step_dirs.sort(key=extract_step)
    return step_dirs[-1]
