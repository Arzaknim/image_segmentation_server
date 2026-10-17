import torch
import segmentation_models_pytorch as smp


if __name__ == "__main__":
    # 1. Recreate model & load .pth weights
    model = smp.Unet(encoder_name="efficientnet-b0", classes=6)
    model.load_state_dict(torch.load("w_pth/efficientnet_b0_tiled.pth", map_location="cpu"))
    model.eval()

    # 2. Export to ONNX
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        model,
        dummy_input,
        "w_onnx/efficientnet_b0_tiled.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )