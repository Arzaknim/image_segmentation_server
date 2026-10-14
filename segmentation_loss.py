import torch
import torch.nn as nn
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
