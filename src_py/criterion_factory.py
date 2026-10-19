#!/usr/bin/env python3
"""Criterion Factory for Semantic Segmentation Losses.

Provides unified creation, configuration, and introspection of segmentation criteria:
  1. CrossEntropyDiceLoss (standard baseline)
  2. LogitFuzzyLoss (fuzzy membership + T-norm Dice + De Luca-Termini entropy)
  3. FocalSoftBCELoss (smp.losses.FocalLoss + smp.losses.SoftBCEWithLogitsLoss + Dice)
  4. FocalLoss (multiclass focal loss directly on logits)
  5. SoftBCEWithLogitsLoss (soft label smoothing BCE on logits)
  6. Standard DiceLoss / CrossEntropyLoss

Keeps training pipelines clean, modular, and configurable via JSON or CLI.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from segmentation_loss import (
    CrossEntropyDiceLoss,
    LogitFuzzyLoss,
    FocalLoss,
    SoftBCEWithLogitsLoss,
    FocalSoftBCELoss,
)


class CriterionFactory:
    """Factory for instantiating and inspecting segmentation loss functions."""

    SUPPORTED_CRITERIA: Dict[str, str] = {
        "ce_dice": "CrossEntropyDiceLoss",
        "cross_entropy_dice": "CrossEntropyDiceLoss",
        "fuzzy": "LogitFuzzyLoss",
        "logit_fuzzy": "LogitFuzzyLoss",
        "focal_bce": "FocalSoftBCELoss",
        "focal_soft_bce": "FocalSoftBCELoss",
        "focal": "FocalLoss",
        "soft_bce": "SoftBCEWithLogitsLoss",
        "dice": "DiceLoss",
        "cross_entropy": "CrossEntropyLoss",
        "ce": "CrossEntropyLoss",
    }

    @classmethod
    def list_supported_criteria(cls) -> List[str]:
        """Return canonical names of supported criteria."""
        return ["ce_dice", "fuzzy", "focal_bce", "focal", "soft_bce", "dice", "ce"]

    @classmethod
    def resolve_criterion_name(cls, name: str) -> str:
        """Resolve criterion alias to canonical name."""
        normalized = name.strip().lower().replace("-", "_")
        if normalized not in cls.SUPPORTED_CRITERIA:
            supported = ", ".join(sorted(cls.SUPPORTED_CRITERIA.keys()))
            raise ValueError(
                f"Unknown criterion '{name}'. Supported criteria: {supported}"
            )
        return cls.SUPPORTED_CRITERIA[normalized]

    @classmethod
    def create_criterion(
        cls,
        criterion_name: str = "ce_dice",
        ignore_index: Optional[int] = 5,
        **kwargs: Any,
    ) -> nn.Module:
        """Instantiate criterion by name or alias with provided parameters."""
        resolved = cls.resolve_criterion_name(criterion_name)

        if resolved == "CrossEntropyDiceLoss":
            return CrossEntropyDiceLoss(ignore_index=ignore_index)

        if resolved == "LogitFuzzyLoss":
            fuzziness = kwargs.get("fuzziness", 0.1)
            fuzzifier_m = kwargs.get("fuzzifier_m", 1.5)
            temperature = kwargs.get("temperature", 1.0)
            entropy_weight = kwargs.get("entropy_weight", 0.0)
            overlap_weight = kwargs.get("overlap_weight", 1.0)
            ce_weight = kwargs.get("ce_weight", 1.0)
            mode = kwargs.get("mode", "dice")
            smooth = kwargs.get("smooth", 1e-5)
            return LogitFuzzyLoss(
                fuzziness=fuzziness,
                fuzzifier_m=fuzzifier_m,
                temperature=temperature,
                entropy_weight=entropy_weight,
                overlap_weight=overlap_weight,
                ce_weight=ce_weight,
                mode=mode,
                smooth=smooth,
                ignore_index=ignore_index,
            )

        if resolved == "FocalSoftBCELoss":
            gamma = kwargs.get("gamma", 2.0)
            smooth_factor = kwargs.get("smooth_factor", 0.1)
            dice_weight = kwargs.get("dice_weight", 1.0)
            bce_weight = kwargs.get("bce_weight", 1.0)
            return FocalSoftBCELoss(
                gamma=gamma,
                smooth_factor=smooth_factor,
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                ignore_index=ignore_index,
            )

        if resolved == "FocalLoss":
            gamma = kwargs.get("gamma", 2.0)
            alpha = kwargs.get("alpha", None)
            return FocalLoss(
                gamma=gamma,
                alpha=alpha,
                ignore_index=ignore_index,
            )

        if resolved == "SoftBCEWithLogitsLoss":
            smooth_factor = kwargs.get("smooth_factor", 0.1)
            pos_weight = kwargs.get("pos_weight", None)
            return SoftBCEWithLogitsLoss(
                smooth_factor=smooth_factor,
                pos_weight=pos_weight,
                ignore_index=ignore_index,
            )

        if resolved == "DiceLoss":
            mode = kwargs.get("mode", "multiclass")
            from_logits = kwargs.get("from_logits", True)
            return smp.losses.DiceLoss(
                mode=mode,
                from_logits=from_logits,
                ignore_index=ignore_index,
            )

        if resolved == "CrossEntropyLoss":
            label_smoothing = kwargs.get("label_smoothing", 0.0)
            return (
                nn.CrossEntropyLoss(
                    ignore_index=ignore_index, label_smoothing=label_smoothing
                )
                if ignore_index is not None
                else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            )

        raise ValueError(f"Unhandled resolved criterion: {resolved}")

    @classmethod
    def from_config(
        cls,
        config: Union[str, Dict[str, Any]],
        default_ignore_index: Optional[int] = 5,
    ) -> nn.Module:
        """Create criterion from a dictionary config or string name."""
        if isinstance(config, str):
            return cls.create_criterion(
                criterion_name=config,
                ignore_index=default_ignore_index,
            )

        if not isinstance(config, dict):
            raise TypeError(f"Criterion config must be dict or str, got {type(config)}")

        cfg = dict(config)
        name = cfg.pop("name", cfg.pop("type", "ce_dice"))
        ignore_index = cfg.pop("ignore_index", default_ignore_index)
        return cls.create_criterion(
            criterion_name=name,
            ignore_index=ignore_index,
            **cfg,
        )

    @classmethod
    def get_criterion_info(
        cls,
        criterion: nn.Module,
        criterion_name: Optional[str] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return structured metadata dictionary for training logs."""
        class_name = criterion.__class__.__name__
        info: Dict[str, Any] = {
            "name": class_name,
            "loss_type": criterion_name or class_name,
            "ignore_index": ignore_index,
        }

        # Extract specific hyperparameter attributes if present
        for attr in (
            "fuzziness",
            "fuzzifier_m",
            "temperature",
            "entropy_weight",
            "gamma",
            "smooth_factor",
            "dice_weight",
            "bce_weight",
        ):
            if hasattr(criterion, attr):
                info[attr] = getattr(criterion, attr)

        for key, val in kwargs.items():
            if key not in info and val is not None:
                info[key] = val

        return info

