import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.cross_entropy = (
            nn.CrossEntropyLoss()
            if ignore_index is None
            else nn.CrossEntropyLoss(ignore_index=ignore_index)
        )
        self.dice = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, targets) + self.dice(logits, targets)


class FuzzyCrossEntropyLogitLoss(nn.Module):
    """Fuzzy Cross-Entropy Loss computed stably from raw network logits.

    In fuzzy set theory for semantic segmentation, ground-truth boundaries
    and mixed pixels possess partial membership across classes. This loss:
    1. Transforms hard one-hot targets into fuzzy membership distributions
       via a fuzziness parameter alpha (label smoothing/mixed-pixel relaxation).
    2. Uses log_softmax on scaled logits (Log-Sum-Exp formulation) for numerical
       stability, avoiding log(0) and gradient explosion.
    """

    def __init__(
        self,
        fuzziness: float = 0.1,
        temperature: float = 1.0,
        ignore_index: int | None = None,
    ):
        super().__init__()
        if not (0.0 <= fuzziness < 1.0):
            raise ValueError("fuzziness must be in [0.0, 1.0)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.fuzziness = fuzziness
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W]
        # targets: [B, H, W]
        B, C, H, W = logits.shape
        scaled_logits = logits / self.temperature
        log_probs = F.log_softmax(scaled_logits, dim=1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            clamped_targets = targets.clone()
            clamped_targets[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            clamped_targets = targets

        if not valid_mask.any():
            return logits.sum() * 0.0

        one_hot = F.one_hot(clamped_targets, num_classes=C).permute(0, 3, 1, 2).float()

        # Fuzzified target membership distribution:
        # mu_target(c) = (1 - alpha) * 1_{y=c} + alpha / C
        if self.fuzziness > 0.0:
            target_membership = (1.0 - self.fuzziness) * one_hot + (self.fuzziness / C)
        else:
            target_membership = one_hot

        valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, C, -1, -1)
        loss = -(target_membership * log_probs)[valid_mask_expanded].sum() / valid_mask.sum()
        return loss


class FuzzyJaccardDiceLogitLoss(nn.Module):
    """Fuzzy T-Norm Overlap (Dice / Jaccard) Loss computed from raw logits.

    Uses fuzzy membership degrees mu = softmax(logits / temperature) and a fuzzifier
    exponent m >= 1.0 (analogous to the weighting exponent in Fuzzy C-Means u_i^m).
    Evaluates continuous fuzzy set intersection (algebraic product T-norm) and
    union (algebraic sum S-norm) directly on the probability space derived from logits.
    """

    def __init__(
        self,
        mode: str = "dice",  # 'dice' or 'jaccard'
        fuzzifier_m: float = 1.5,
        temperature: float = 1.0,
        smooth: float = 1e-5,
        ignore_index: int | None = None,
    ):
        super().__init__()
        if mode not in ("dice", "jaccard"):
            raise ValueError(f"mode must be 'dice' or 'jaccard', got {mode}")
        if fuzzifier_m < 1.0:
            raise ValueError("fuzzifier_m must be >= 1.0")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.mode = mode
        self.fuzzifier_m = fuzzifier_m
        self.temperature = temperature
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits / self.temperature, dim=1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            clamped_targets = targets.clone()
            clamped_targets[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            clamped_targets = targets

        if not valid_mask.any():
            return logits.sum() * 0.0

        one_hot = F.one_hot(clamped_targets, num_classes=C).permute(0, 3, 1, 2).float()

        # Fuzzified membership: mu^m (FCM weighting principle)
        u_m = probs**self.fuzzifier_m

        mask_expanded = valid_mask.unsqueeze(1).float()
        u_m = u_m * mask_expanded
        targets_masked = one_hot * mask_expanded

        # Flatten spatial dimensions per class: [B, C, H*W]
        u_flat = u_m.view(B, C, -1)
        t_flat = targets_masked.view(B, C, -1)

        # Fuzzy T-norm intersection: sum(u^m * target)
        intersection = (u_flat * t_flat).sum(dim=-1)

        if self.mode == "dice":
            cardinality = u_flat.sum(dim=-1) + t_flat.sum(dim=-1)
            score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        else:  # jaccard
            union = u_flat.sum(dim=-1) + t_flat.sum(dim=-1) - intersection
            score = (intersection + self.smooth) / (union + self.smooth)

        if self.ignore_index is not None and 0 <= self.ignore_index < C:
            active_classes = [c for c in range(C) if c != self.ignore_index]
            score = score[:, active_classes]

        return (1.0 - score).mean()


class LogitFuzzyLoss(nn.Module):
    """Composite Custom Fuzzy Segmentation Loss using Logits.

    Combines:
    1. Fuzzy Cross-Entropy with soft target membership (fuzziness parameter)
    2. Fuzzy T-Norm Overlap Loss (with FCM fuzzifier exponent m)
    3. Optional Fuzzy Entropy Regularization (De Luca-Termini measure of fuzziness)
       to penalize excessive ambiguity outside boundary transition regions.

    Drop-in replacement for CrossEntropyDiceLoss.
    """

    def __init__(
        self,
        fuzziness: float = 0.1,
        fuzzifier_m: float = 1.5,
        temperature: float = 1.0,
        entropy_weight: float = 0.0,
        overlap_weight: float = 1.0,
        ce_weight: float = 1.0,
        mode: str = "dice",
        smooth: float = 1e-5,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.overlap_weight = overlap_weight
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.ignore_index = ignore_index

        self.fuzzy_ce = FuzzyCrossEntropyLogitLoss(
            fuzziness=fuzziness,
            temperature=temperature,
            ignore_index=ignore_index,
        )
        self.fuzzy_overlap = FuzzyJaccardDiceLogitLoss(
            mode=mode,
            fuzzifier_m=fuzzifier_m,
            temperature=temperature,
            smooth=smooth,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = logits.new_tensor(0.0)
        if self.ce_weight > 0.0:
            loss = loss + self.ce_weight * self.fuzzy_ce(logits, targets)
        if self.overlap_weight > 0.0:
            loss = loss + self.overlap_weight * self.fuzzy_overlap(logits, targets)

        if self.entropy_weight > 0.0:
            # De Luca-Termini Fuzzy Entropy on logits
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
            log_probs = F.log_softmax(scaled_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                entropy_loss = entropy[valid_mask].mean() if valid_mask.any() else logits.new_tensor(0.0)
            else:
                entropy_loss = entropy.mean()
            loss = loss + self.entropy_weight * entropy_loss

        return loss


class SoftBCEWithLogitsLoss(nn.Module):
    """Wrapper around smp.losses.SoftBCEWithLogitsLoss for multi-class target adaptation.

    Delegates calculation directly to segmentation_models_pytorch.losses.SoftBCEWithLogitsLoss.
    Adapts 2D integer class masks [B, H, W] from the DataLoader into one-hot float
    tensors [B, C, H, W] required by SMP's SoftBCEWithLogitsLoss.
    """

    def __init__(
        self,
        smooth_factor: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        ignore_index: int | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss(
            smooth_factor=smooth_factor,
            pos_weight=pos_weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            C = logits.shape[1]
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                clamped_targets = targets.clone()
                clamped_targets[~valid_mask] = 0
                targets_one_hot = F.one_hot(clamped_targets, num_classes=C).permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
            return self.loss_fn(logits, targets_one_hot)
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Multi-Class Focal Loss delegating directly to smp.losses.FocalLoss."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | None = None,
        ignore_index: int | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_fn = smp.losses.FocalLoss(
            mode="multiclass",
            gamma=gamma,
            alpha=alpha,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalSoftBCELoss(nn.Module):
    """Composite loss built using smp.losses.FocalLoss and smp.losses.SoftBCEWithLogitsLoss.

    Combines:
    1. smp.losses.FocalLoss(mode='multiclass')
    2. smp.losses.SoftBCEWithLogitsLoss()
    3. smp.losses.DiceLoss(mode='multiclass') (optional)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        smooth_factor: float = 0.1,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal = smp.losses.FocalLoss(
            mode="multiclass",
            gamma=gamma,
            ignore_index=ignore_index,
        )
        self.soft_bce = SoftBCEWithLogitsLoss(
            smooth_factor=smooth_factor,
            ignore_index=ignore_index,
        )
        self.dice = (
            smp.losses.DiceLoss(
                mode="multiclass",
                from_logits=True,
                ignore_index=ignore_index,
            )
            if dice_weight > 0.0
            else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.focal(logits, targets) + self.bce_weight * self.soft_bce(logits, targets)
        if self.dice is not None:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        return loss
