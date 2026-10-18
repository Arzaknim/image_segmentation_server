#!/usr/bin/env python3
"""
Model Factory for Aerial Semantic Segmentation.

Provides unified creation, configuration, and export preparation for:
  1. EfficientNet-B0 (U-Net) baseline
  2. SegFormer (MiT-B0, MiT-B1, etc.) Vision Transformers
  3. FastViT (FastViT-T8, FastViT-SA12, etc.) Reparameterized Hybrid Transformers
  4. Extensible SMP / timm architectures

Keeps training scripts clean and modular while enabling seamless model swapping.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("model_factory")

# Try importing segmentation_models_pytorch and timm
try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ==============================================================================
# FastViT Segmentation Model with Lightweight MLP Decoder
# ==============================================================================

class SegFormerMLPHead(nn.Module):
    """
    Lightweight all-MLP segmentation decoder inspired by SegFormer.
    Projects multi-scale hierarchical feature stages to a unified embedding dimension,
    upsamples to 1/4 resolution, fuses with 1x1 convolution, and predicts class logits.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        embedding_dim: int = 256,
        num_classes: int = 6,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Linear projection for each stage
        self.linear_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embedding_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            for dim in in_channels_list
        ])

        # Feature fusion block
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels_list), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of 4 feature maps [C1, C2, C3, C4] from stages 1..4.
        Returns:
            Logits at 1/4 input spatial resolution.
        """
        c1, c2, c3, c4 = features
        h4, w4 = c1.shape[2], c1.shape[3]

        p1 = self.linear_projections[0](c1)
        p2 = F.interpolate(self.linear_projections[1](c2), size=(h4, w4), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.linear_projections[2](c3), size=(h4, w4), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.linear_projections[3](c4), size=(h4, w4), mode="bilinear", align_corners=False)

        fused = self.linear_fuse(torch.cat([p1, p2, p3, p4], dim=1))
        fused = self.dropout(fused)
        return self.classifier(fused)


class FastViTSegmentation(nn.Module):
    """
    FastViT Hybrid Vision Transformer for Semantic Segmentation.
    Combines an Apple FastViT backbone with a lightweight SegFormer MLP decoder head.
    Supports structural reparameterization (`reparameterize()`) for fast inference and clean ONNX export.
    """

    def __init__(
        self,
        backbone: str = "fastvit_t8",
        num_classes: int = 6,
        in_channels: int = 3,
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout_ratio: float = 0.1,
    ):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required to build FastViT models. Run: pip install timm")

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # Instantiate hierarchical feature extraction backbone
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
        )

        encoder_channels = [info["num_chs"] for info in self.encoder.feature_info]
        self.decoder = SegFormerMLPHead(
            in_channels_list=encoder_channels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = (x.shape[2], x.shape[3])
        features = self.encoder(x)
        logits_low_res = self.decoder(features)
        return F.interpolate(logits_low_res, size=input_size, mode="bilinear", align_corners=False)

    def reparameterize(self) -> "FastViTSegmentation":
        """
        Collapses all multi-branch RepMixer and ReparamLargeKernel convolutions
        into single dense 3x3 kernels for maximal inference speed and clean ONNX export.
        """
        self.eval()
        count = 0
        for mod in self.modules():
            if hasattr(mod, "reparameterize") and mod is not self:
                try:
                    mod.reparameterize()
                    count += 1
                except Exception as err:
                    logger.debug("Submodule %s reparameterization skipped: %s", mod, err)
        logger.info("Successfully reparameterized %d submodules in FastViT.", count)
        return self


# ==============================================================================
# Model Factory
# ==============================================================================

MODEL_REGISTRY = {
    # 1. Baseline U-Net EfficientNet
    "efficientnet_b0": {
        "family": "smp_unet",
        "encoder": "efficientnet-b0",
        "default_weights": "imagenet",
        "description": "Baseline U-Net with EfficientNet-B0 backbone (5.3M params)",
    },
    "efficientnet_b2": {
        "family": "smp_unet",
        "encoder": "efficientnet-b2",
        "default_weights": "imagenet",
        "description": "U-Net with EfficientNet-B2 backbone (9.1M params)",
    },
    "efficientnet_b3": {
        "family": "smp_unet",
        "encoder": "efficientnet-b3",
        "default_weights": "imagenet",
        "description": "U-Net with EfficientNet-B3 backbone (12.2M params)",
    },
    # 2. SegFormer (Pure Hierarchical Vision Transformer)
    "segformer_b0": {
        "family": "smp_segformer",
        "encoder": "mit_b0",
        "default_weights": "imagenet",
        "description": "SegFormer with MiT-B0 backbone + MLP decoder (3.7M params, INT8 GEMM friendly)",
    },
    "segformer_b1": {
        "family": "smp_segformer",
        "encoder": "mit_b1",
        "default_weights": "imagenet",
        "description": "SegFormer with MiT-B1 backbone + MLP decoder (13.7M params, high aerial mIoU)",
    },
    "segformer_b2": {
        "family": "smp_segformer",
        "encoder": "mit_b2",
        "default_weights": "imagenet",
        "description": "SegFormer with MiT-B2 backbone + MLP decoder (24.7M params)",
    },
    # 3. FastViT (Structural Reparameterization + Hybrid Vision Transformer)
    "fastvit_t8": {
        "family": "fastvit",
        "encoder": "fastvit_t8",
        "default_weights": True,
        "description": "FastViT-T8 hybrid backbone + SegFormer MLP decoder (4.0M params, ultra-fast latency)",
    },
    "fastvit_sa12": {
        "family": "fastvit",
        "encoder": "fastvit_sa12",
        "default_weights": True,
        "description": "FastViT-SA12 self-attention hybrid backbone + MLP decoder (11.5M params, high mIoU)",
    },
    "fastvit_s12": {
        "family": "fastvit",
        "encoder": "fastvit_s12",
        "default_weights": True,
        "description": "FastViT-S12 convolution-focused hybrid backbone + MLP decoder (8.8M params)",
    },
    "fastvit_sa24": {
        "family": "fastvit",
        "encoder": "fastvit_sa24",
        "default_weights": True,
        "description": "FastViT-SA24 deep self-attention hybrid backbone + MLP decoder (21.5M params)",
    },
}

# Aliases
MODEL_ALIASES = {
    "unet": "efficientnet_b0",
    "efficientnet": "efficientnet_b0",
    "efnb0": "efficientnet_b0",
    "segformer": "segformer_b0",
    "mit_b0": "segformer_b0",
    "mit_b1": "segformer_b1",
    "fastvit": "fastvit_t8",
}


class ModelFactory:
    """
    Factory class to instantiate and configure semantic segmentation models.
    """

    @staticmethod
    def list_models() -> List[Dict[str, str]]:
        """Returns list of supported model configurations."""
        return [
            {
                "name": name,
                "family": meta["family"],
                "encoder": meta["encoder"],
                "description": meta["description"],
            }
            for name, meta in MODEL_REGISTRY.items()
        ]

    @staticmethod
    def resolve_model_name(name_or_alias: str) -> str:
        key = name_or_alias.lower().strip().replace("-", "_")
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        if key in MODEL_REGISTRY:
            return key
        # Partial prefix matching (e.g. 'segformer_b0' or 'fastvit')
        for k in MODEL_REGISTRY:
            if key == k:
                return k
        return key

    @classmethod
    def create_model(
        cls,
        model_name: str = "efficientnet_b0",
        num_classes: int = 6,
        in_channels: int = 3,
        pretrained: bool = True,
        encoder_name: Optional[str] = None,
        embedding_dim: int = 256,
        dropout_ratio: float = 0.1,
    ) -> nn.Module:
        """
        Creates and returns a segmentation PyTorch model.

        Args:
            model_name: Name or alias of the model (e.g. 'efficientnet_b0', 'segformer_b0', 'fastvit_t8').
            num_classes: Number of segmentation output classes (default: 6 for Dubai dataset).
            in_channels: Number of input image channels (default: 3 for RGB).
            pretrained: Whether to load ImageNet pretrained weights for the encoder.
            encoder_name: Optional override for the backbone encoder name.
            embedding_dim: Embedding dimension for MLP decoders (FastViT/SegFormer).
            dropout_ratio: Dropout probability before classifier.

        Returns:
            nn.Module: Instantiated PyTorch segmentation model.
        """
        canonical_name = cls.resolve_model_name(model_name)

        # 1. Known Registry Models
        if canonical_name in MODEL_REGISTRY:
            meta = MODEL_REGISTRY[canonical_name]
            family = meta["family"]
            enc = encoder_name or meta["encoder"]

            if family == "smp_unet":
                if not HAS_SMP:
                    raise ImportError("segmentation_models_pytorch is required. Run: pip install segmentation-models-pytorch")
                weights = meta["default_weights"] if pretrained else None
                logger.info("Building SMP U-Net with encoder '%s' (pretrained=%s)...", enc, weights)
                return smp.Unet(
                    encoder_name=enc,
                    encoder_weights=weights,
                    in_channels=in_channels,
                    classes=num_classes,
                )

            elif family == "smp_segformer":
                if not HAS_SMP:
                    raise ImportError("segmentation_models_pytorch is required. Run: pip install segmentation-models-pytorch")
                weights = meta["default_weights"] if pretrained else None
                logger.info("Building SMP SegFormer with encoder '%s' (pretrained=%s)...", enc, weights)
                return smp.Segformer(
                    encoder_name=enc,
                    encoder_weights=weights,
                    in_channels=in_channels,
                    classes=num_classes,
                )

            elif family == "fastvit":
                logger.info("Building FastViTSegmentation with backbone '%s' (pretrained=%s)...", enc, pretrained)
                return FastViTSegmentation(
                    backbone=enc,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    embedding_dim=embedding_dim,
                    pretrained=pretrained,
                    dropout_ratio=dropout_ratio,
                )

        # 2. Direct SegFormer Fallback (if user passed 'segformer' with custom mit_* encoder)
        if "segformer" in canonical_name or canonical_name.startswith("mit_"):
            if not HAS_SMP:
                raise ImportError("segmentation_models_pytorch is required.")
            enc = encoder_name or (canonical_name if canonical_name.startswith("mit_") else "mit_b0")
            weights = "imagenet" if pretrained else None
            logger.info("Building SegFormer fallback with encoder '%s'...", enc)
            return smp.Segformer(
                encoder_name=enc,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=num_classes,
            )

        # 3. Direct FastViT Fallback (if user passed a fastvit_* name from timm)
        if "fastvit" in canonical_name:
            enc = encoder_name or canonical_name
            logger.info("Building FastViT fallback with backbone '%s'...", enc)
            return FastViTSegmentation(
                backbone=enc,
                num_classes=num_classes,
                in_channels=in_channels,
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                dropout_ratio=dropout_ratio,
            )

        # 4. Generic SMP U-Net Fallback
        if HAS_SMP:
            logger.warning("Unrecognized model '%s'. Falling back to smp.Unet with encoder '%s'.", model_name, canonical_name)
            weights = "imagenet" if pretrained else None
            return smp.Unet(
                encoder_name=canonical_name,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=num_classes,
            )

        raise ValueError(f"Unsupported model name: '{model_name}'. Run ModelFactory.list_models() to see supported models.")

    @classmethod
    def get_model_info(
        cls,
        model: nn.Module,
        model_name: str,
        num_classes: int = 6,
        in_channels: int = 3,
        pretrained: bool = True,
    ) -> Dict[str, Any]:
        """Returns structured metadata for training logs."""
        canonical_name = cls.resolve_model_name(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if canonical_name in MODEL_REGISTRY:
            meta = MODEL_REGISTRY[canonical_name]
            architecture = meta["family"]
            encoder = meta["encoder"]
        elif isinstance(model, FastViTSegmentation):
            architecture = "FastViTSegmentation"
            encoder = model.backbone_name
        elif hasattr(model, "name") and "segformer" in getattr(model, "name", "").lower():
            architecture = "Segformer"
            encoder = getattr(model, "encoder_name", "mit_b0")
        else:
            architecture = model.__class__.__name__
            encoder = getattr(model, "encoder_name", canonical_name)

        return {
            "name": canonical_name,
            "architecture": architecture,
            "encoder": encoder,
            "pretrained": pretrained,
            "input_channels": in_channels,
            "classes": num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "has_reparameterize": hasattr(model, "reparameterize"),
        }

    @staticmethod
    def prepare_for_inference_or_export(model: nn.Module) -> nn.Module:
        """
        Prepares model for evaluation, ONNX export, or quantization.
        Automatically triggers structural reparameterization if available.
        """
        model.eval()
        if hasattr(model, "reparameterize") and callable(getattr(model, "reparameterize")):
            model.reparameterize()
        return model


# Module-level convenience functions
def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 6,
    in_channels: int = 3,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Convenience wrapper for ModelFactory.create_model."""
    return ModelFactory.create_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model creation via ModelFactory")
    parser.add_argument("-m", "--model", default="fastvit_t8", choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    print(f"Creating model: {args.model}")
    m = ModelFactory.create_model(args.model, num_classes=6)
    info = ModelFactory.get_model_info(m, args.model)
    print("Model Metadata:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    dummy = torch.randn(1, 3, 512, 512)
    m = ModelFactory.prepare_for_inference_or_export(m)
    with torch.no_grad():
        out = m(dummy)
    print(f"Forward pass output shape: {out.shape}")

