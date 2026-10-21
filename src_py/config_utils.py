#!/usr/bin/env python3
"""Configuration Loader and Validator for Semantic Segmentation Training.

Supports loading hyperparameters from JSON configs with support for both:
  1. Flat configurations (e.g. {"model": "efficientnet_b0", "epochs": 40, "batch_size": 6})
  2. Hierarchical configurations (e.g. {"model": {"name": "..."}, "training": {"num_epochs": 40}})

Provides merging between JSON config defaults and explicit CLI overrides.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a JSON or JSONC configuration file with comments."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip single-line (//) and multi-line (/* */) comments to allow annotated templates
    clean_lines = []
    for line in content.splitlines():
        # Strip trailing single-line comment if not inside string
        clean_lines.append(re.sub(r"\s*//.*$", "", line))
    clean_content = "\n".join(clean_lines)
    clean_content = re.sub(r"/\*.*?\*/", "", clean_content, flags=re.DOTALL)

    data = json.loads(clean_content)

    if not isinstance(data, dict):
        raise ValueError(f"Root configuration must be a JSON object, got {type(data)}")

    return data


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize nested or flat configuration dictionary into standard flat keys."""
    flat: Dict[str, Any] = {}

    for key, val in config.items():
        if key.startswith("_"):
            continue

        if isinstance(val, dict):
            # Inspect nested section
            if key in ("model", "architecture"):
                if "name" in val:
                    flat["model_name"] = val["name"]
                if "pretrained" in val:
                    flat["pretrained"] = val["pretrained"]
                if "checkpoint_path" in val:
                    flat["checkpoint_path"] = val["checkpoint_path"]
            elif key in ("dataset", "data"):
                for subk, subv in val.items():
                    if subk.startswith("_"):
                        continue
                    norm_k = "tile_stride" if subk == "stride" else "dataset_dir" if subk == "dataset_base_dir" else subk
                    flat[norm_k] = subv
            elif key in ("training", "train"):
                for subk in ("num_epochs", "epochs", "batch_size", "device", "eval_every", "log_dir"):
                    if subk in val:
                        norm_k = "epochs" if subk == "num_epochs" else subk
                        flat[norm_k] = val[subk]
            elif key in ("optimizer", "opt"):
                flat["optimizer"] = val
                if "lr" in val or "learning_rate" in val:
                    flat["lr"] = val.get("lr", val.get("learning_rate"))
                if "weight_decay" in val:
                    flat["weight_decay"] = val["weight_decay"]
            elif key in ("criterion", "loss"):
                flat["criterion"] = val
                flat["loss"] = val.get("name", val.get("type", "ce_dice"))
            elif key in ("scheduler", "sched"):
                flat["scheduler"] = val
            else:
                if not key.startswith("_"):
                    flat[key] = val
        else:
            if key.startswith("_"):
                continue
            # Direct flat assignment
            norm_k = (
                "model_name"
                if key in ("model", "architecture")
                else "epochs"
                if key == "num_epochs"
                else "tile_stride"
                if key == "stride"
                else "dataset_dir"
                if key == "dataset_base_dir"
                else key
            )
            flat[norm_k] = val

    return flat


def merge_config_with_args(
    config: Optional[Dict[str, Any]],
    args_dict: Dict[str, Any],
    explicit_cli_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """Merge JSON config with command-line arguments.

    Values explicitly set on the CLI take precedence over config files.
    Config file values take precedence over default CLI argument values.
    """
    merged = dict(args_dict)

    if not config:
        return merged

    normalized_cfg = flatten_config(config)

    for k, v in normalized_cfg.items():
        # If CLI user did not explicitly provide this key, use config's value
        if explicit_cli_keys is None or k not in explicit_cli_keys:
            merged[k] = v

    return merged

