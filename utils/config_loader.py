"""
Configuration loader for LlamaGen with YAML support.

This module provides utilities to load YAML configurations with:
- Hierarchical config composition
- CLI argument override support
- OmegaConf integration
"""

import os
import yaml
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
import argparse


def load_config(config_path: str) -> DictConfig:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        OmegaConf DictConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return OmegaConf.create(config_dict)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configs with later configs overriding earlier ones.

    Args:
        *configs: Variable number of OmegaConf configs

    Returns:
        Merged OmegaConf config
    """
    if not configs:
        return OmegaConf.create({})

    merged = configs[0]
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)

    return merged


def args_to_config(args: argparse.Namespace) -> DictConfig:
    """
    Convert argparse Namespace to OmegaConf config.

    Args:
        args: Argparse Namespace object

    Returns:
        OmegaConf config
    """
    args_dict = vars(args)
    return OmegaConf.create(args_dict)


def override_config_from_args(
    config: DictConfig, args: argparse.Namespace
) -> DictConfig:
    """
    Override config values with CLI arguments.
    CLI arguments have the highest priority.

    Args:
        config: Base OmegaConf config
        args: CLI arguments

    Returns:
        Config with overrides applied
    """
    # Convert args to config
    args_config = args_to_config(args)

    # Filter out None values from args (not provided on CLI)
    args_dict = OmegaConf.to_container(args_config, resolve=True)
    args_dict_filtered = {k: v for k, v in args_dict.items() if v is not None}

    # Apply overrides
    overrides = OmegaConf.create(args_dict_filtered)
    config = OmegaConf.merge(config, overrides)

    return config


def load_config_with_cli_override(
    config_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    base_config: Optional[str] = None,
) -> DictConfig:
    """
    Load config with CLI override support.

    Priority (highest to lowest):
    1. CLI arguments (--arg value)
    2. Config file specified via --config
    3. Base config (defaults)

    Args:
        config_path: Path to config file (from --config arg)
        args: CLI arguments
        base_config: Path to base/default config

    Returns:
        Final merged config
    """
    configs = []

    # 1. Load base config if provided
    if base_config and os.path.exists(base_config):
        configs.append(load_config(base_config))

    # 2. Load user config if provided
    if config_path and os.path.exists(config_path):
        configs.append(load_config(config_path))

    # 3. Merge configs
    if configs:
        merged_config = merge_configs(*configs)
    else:
        merged_config = OmegaConf.create({})

    # 4. Override with CLI args (highest priority)
    if args is not None:
        merged_config = override_config_from_args(merged_config, args)

    return merged_config


def config_to_args(config: DictConfig) -> argparse.Namespace:
    """
    Convert OmegaConf config back to argparse Namespace.
    Useful for compatibility with existing training scripts.

    Args:
        config: OmegaConf config

    Returns:
        Argparse Namespace
    """
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Flatten nested dicts with dot notation
    flat_dict = {}

    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, f"{key}.")
            else:
                flat_dict[key] = v

    flatten(config_dict)

    return argparse.Namespace(**flat_dict)


def print_config(config: DictConfig, title: str = "Configuration"):
    """
    Pretty print configuration.

    Args:
        config: OmegaConf config
        title: Title to display
    """
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(OmegaConf.to_yaml(config))
    print(f"{'='*80}\n")


def save_config(config: DictConfig, save_path: str):
    """
    Save config to YAML file.

    Args:
        config: OmegaConf config
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


# Example usage
if __name__ == "__main__":
    # Example: Load a config
    config = load_config("configs/examples/gpt_l_scheduled.yaml")
    print_config(config, "Loaded Config")

    # Example: Override with CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)

    args = parser.parse_args(["--lr", "0.0005", "--epochs", "500"])

    config = override_config_from_args(config, args)
    print_config(config, "Config After CLI Override")
