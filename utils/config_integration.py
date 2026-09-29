"""
Configuration integration utilities for training scripts.

Provides helpers to integrate YAML configs with argparse, ensuring CLI args override config values.
"""

import argparse
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from .config_loader import load_config, merge_configs, override_config_from_args


def add_config_argument(parser: argparse.ArgumentParser):
    """
    Add --config argument to parser.

    Args:
        parser: Argparse parser to add argument to
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )


def merge_config_and_args(
    args: argparse.Namespace,
    base_config_path: Optional[str] = None,
) -> argparse.Namespace:
    """
    Merge YAML config with CLI arguments, with CLI taking priority.

    Priority (highest to lowest):
    1. CLI arguments (if explicitly provided)
    2. Config file (--config path)
    3. Base config (default values)

    Args:
        args: Parsed CLI arguments
        base_config_path: Optional path to base config file

    Returns:
        Namespace with merged values
    """
    # Start with empty config
    config = OmegaConf.create({})

    # 1. Load base config if provided
    if base_config_path:
        config = merge_configs(config, load_config(base_config_path))

    # 2. Load user config if --config is specified
    if hasattr(args, "config") and args.config:
        user_config = load_config(args.config)
        config = merge_configs(config, user_config)

    # 3. Convert config to flat namespace for compatibility
    config_dict = OmegaConf.to_container(config, resolve=True)
    flat_config = flatten_config(config_dict)

    # 4. Create namespace from config
    config_namespace = argparse.Namespace(**flat_config)

    # 5. Override with CLI args (only non-None values)
    args_dict = vars(args)
    for key, value in args_dict.items():
        # Skip 'config' key itself and None values (not provided on CLI)
        if key != "config" and value is not None:
            setattr(config_namespace, key, value)
        elif not hasattr(config_namespace, key):
            # Add CLI args that aren't in config
            setattr(config_namespace, key, value)

    return config_namespace


def flatten_config(config_dict: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Flatten nested config dict.

    Example:
        {'model': {'dim': 1024, 'n_layer': 24}}
        -> {'model_dim': 1024, 'model_n_layer': 24}

    Args:
        config_dict: Nested dictionary
        parent_key: Parent key for recursion
        sep: Separator for flattening

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in config_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_config(flat_dict: dict, sep: str = "_") -> dict:
    """
    Unflatten a flattened config dict.

    Example:
        {'model_dim': 1024, 'model_n_layer': 24}
        -> {'model': {'dim': 1024, 'n_layer': 24}}

    Args:
        flat_dict: Flattened dictionary
        sep: Separator used in flattening

    Returns:
        Nested dictionary
    """
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested


def apply_config_to_args(
    args: argparse.Namespace,
    config_path: Optional[str] = None,
) -> argparse.Namespace:
    """
    Simple helper to apply config to args with CLI override.

    This is the main function to use in training scripts.

    Args:
        args: Parsed CLI arguments (must include --config if used)
        config_path: Optional config path (alternative to args.config)

    Returns:
        Merged arguments with config values

    Example:
        parser = argparse.ArgumentParser()
        # ... add arguments ...
        parser.add_argument("--config", type=str, default=None)
        args = parser.parse_args()

        # Apply config with CLI override
        args = apply_config_to_args(args)

        # Now args has config values, but CLI args take priority
    """
    if config_path:
        args.config = config_path

    if hasattr(args, "config") and args.config:
        config = load_config(args.config)

        # Convert config to flat namespace
        config_dict = OmegaConf.to_container(config, resolve=True)
        flat_config = flatten_config(config_dict)

        # Apply config values to args (only if not already set by CLI)
        for key, value in flat_config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            elif getattr(args, key) is None:
                # If arg is None (not provided on CLI), use config value
                setattr(args, key, value)
            # else: CLI value takes priority, don't override

    return args
