#!/usr/bin/env python3
"""
Alias / Convenience module for ONNX quantization.
Re-exports all functionality from src_py.quantize_onnx.
"""

from src_py.quantize_onnx import (
    AerialCalibrationDataReader,
    DEFAULT_DYNAMIC_OUTPUT_MODEL,
    DEFAULT_INPUT_MODEL,
    DEFAULT_STATIC_OUTPUT_MODEL,
    find_default_calibration_images,
    main,
    preprocess_image,
    preprocess_onnx_graph,
    quantize_onnx,
    quantize_onnx_dynamic,
    quantize_onnx_static,
    verify_and_benchmark,
)

if __name__ == "__main__":
    main()

