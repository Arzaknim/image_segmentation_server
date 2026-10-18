#!/usr/bin/env python3
"""
ONNX Model Export Script for Aerial Semantic Segmentation.
Exports trained PyTorch weights to production ONNX format with dynamic batching.
Supports EfficientNet, SegFormer, and FastViT via ModelFactory.
"""

import argparse
import os
import torch
from model_factory import ModelFactory


def export_to_onnx(
    model_name: str = "efficientnet_b0",
    weights_path: str = "w_pth/efficientnet_b0_tiled.pth",
    output_onnx_path: str = "w_onnx/efficientnet_b0_tiled.onnx",
    num_classes: int = 6,
    tile_size: int = 512,
    opset_version: int = 17,
):
    print(f"Loading '{model_name}' model architecture...")
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    if os.path.isfile(weights_path):
        print(f"Loading weights from '{weights_path}'...")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint '{weights_path}' not found. Exporting uninitialized/random model for testing.")

    # Prepare for export (automatically triggers reparameterization for FastViT)
    model = ModelFactory.prepare_for_inference_or_export(model)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    dummy_input = torch.randn(1, 3, tile_size, tile_size)

    print(f"Exporting ONNX graph to '{output_onnx_path}' (Opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=opset_version,
    )
    print(f"ONNX export successful: {output_onnx_path} ({os.path.getsize(output_onnx_path) / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch segmentation checkpoint to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="efficientnet_b0",
        help="Model architecture: 'efficientnet_b0', 'segformer_b0', 'fastvit_t8', etc.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="Path to .pth checkpoint file. Defaults to w_pth/{model}_tiled.pth",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for exported .onnx model. Defaults to w_onnx/{model}_tiled.onnx",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Spatial resolution of dummy input")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    canonical = ModelFactory.resolve_model_name(args.model)
    weights = args.weights or f"w_pth/{canonical}_tiled.pth"
    output = args.output or f"w_onnx/{canonical}_tiled.onnx"

    export_to_onnx(
        model_name=args.model,
        weights_path=weights,
        output_onnx_path=output,
        tile_size=args.tile_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
    print(f"ONNX export successful: {output_onnx_path} ({os.path.getsize(output_onnx_path) / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch segmentation checkpoint to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="efficientnet_b0",
        help="Model architecture: 'efficientnet_b0', 'segformer_b0', 'fastvit_t8', etc.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="Path to .pth checkpoint file. Defaults to w_pth/{model}_tiled.pth",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for exported .onnx model. Defaults to w_onnx/{model}_tiled.onnx",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Spatial resolution of dummy input")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    canonical = ModelFactory.resolve_model_name(args.model)
    weights = args.weights or f"w_pth/{canonical}_tiled.pth"
    output = args.output or f"w_onnx/{canonical}_tiled.onnx"

    export_to_onnx(
        model_name=args.model,
        weights_path=weights,
        output_onnx_path=output,
        tile_size=args.tile_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
    )