#!/usr/bin/env python3
"""
ONNX INT8 Quantization Pipeline for Aerial Imagery Semantic Segmentation.

This module provides tools to quantize full-precision (FP32) ONNX segmentation models
to lower precision (INT8/UINT8), supporting:
  1. Dynamic Quantization: Quantizes weight tensors ahead-of-time without requiring
     a calibration dataset. Activations are dynamically quantized during execution.
  2. Static Quantization (QDQ / QOperator): Quantizes both weights and activations using
     a calibration dataset (e.g. aerial tiles with ImageNet normalization).
  3. Pre-quantization graph optimization and shape inference via `quant_pre_process`.
  4. Accuracy & Performance Verification: Compares FP32 vs INT8 logits, IoU/prediction
     agreement percentage, latency speedup, and disk size reduction.

Usage:
    # Run dynamic quantization (fast, no calibration dataset required):
    python src_py/quantize_onnx.py --mode dynamic

    # Run static quantization with calibration images:
    python src_py/quantize_onnx.py --mode static --calibrate-data Dataset_Tiled/val/images/ --num-samples 32

    # Quantize and compare both modes with comprehensive verification:
    python src_py/quantize_onnx.py --mode both --verify
"""

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quantize_onnx")

# Check onnx & onnxruntime availability
try:
    import onnx
    from onnx import checker
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError as err:
    logger.error(
        "Missing required dependencies: %s.\n"
        "Please ensure 'onnx' and 'onnxruntime' are installed in your environment.\n"
        "Example: conda activate image_seg OR pip install onnx onnxruntime",
        err,
    )
    sys.exit(1)

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ==============================================================================
# Constants & Defaults
# ==============================================================================

DEFAULT_INPUT_MODEL = "w_onnx/efficientnet_b0_tiled.onnx"
DEFAULT_DYNAMIC_OUTPUT_MODEL = "w_onnx/efficientnet_b0_tiled_dynamic_int8.onnx"
DEFAULT_STATIC_OUTPUT_MODEL = "w_onnx/efficientnet_b0_tiled_static_int8.onnx"

DEFAULT_TILE_SIZE = 512
DEFAULT_NUM_CLASSES = 6
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CANDIDATE_DATASET_DIRS = [
    "Dataset_Tiled/val/images",
    "Dataset_Tiled/train/images",
    "Dataset_Flattened/images",
    "../Dataset_Tiled/val/images",
    "../Dataset_Tiled/train/images",
    "../Dataset_Flattened/images",
]


# ==============================================================================
# Image Preprocessing & Calibration Data Reader
# ==============================================================================

def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE),
) -> np.ndarray:
    """
    Loads and preprocesses an aerial image matching the model's expected input signature:
      1. Converts to RGB.
      2. Resizes to target_size (height, width).
      3. Scales pixel values to [0.0, 1.0].
      4. Normalizes with ImageNet mean and std.
      5. Transposes from HWC to CHW and adds batch dimension (1, 3, H, W).
    """
    path_str = str(image_path)
    if HAS_OPENCV:
        img_bgr = cv2.imread(path_str)
        if img_bgr is None:
            raise ValueError(f"Failed to load image from '{path_str}' via OpenCV")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != target_size:
            img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        img_float = img_rgb.astype(np.float32) / 255.0
    elif HAS_PIL:
        with Image.open(path_str) as pil_img:
            pil_rgb = pil_img.convert("RGB")
            if pil_rgb.size != (target_size[1], target_size[0]):
                pil_rgb = pil_rgb.resize((target_size[1], target_size[0]), Image.BILINEAR)
            img_float = np.array(pil_rgb, dtype=np.float32) / 255.0
    else:
        raise RuntimeError("Neither OpenCV (cv2) nor Pillow (PIL) is installed to read calibration images.")

    # ImageNet normalization
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW -> NCHW (1, 3, H, W)
    chw = np.transpose(img_norm, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)


def find_default_calibration_images(limit: int = 50) -> List[str]:
    """Finds available aerial calibration images across standard repository folders."""
    for folder in CANDIDATE_DATASET_DIRS:
        if os.path.isdir(folder):
            patterns = [os.path.join(folder, "*.png"), os.path.join(folder, "*.jpg"), os.path.join(folder, "*.jpeg")]
            files: List[str] = []
            for pat in patterns:
                files.extend(glob.glob(pat))
            if files:
                files.sort()
                logger.info("Auto-detected %d calibration images in '%s'", len(files), folder)
                return files[:limit]
    return []


class AerialCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for static quantization of the Dubai Aerial Segmentation ONNX model.
    Feeds normalized NCHW (1, 3, 512, 512) float32 tensors to ONNX Runtime quantizer.
    """

    def __init__(
        self,
        image_paths: Optional[List[Union[str, Path]]] = None,
        input_name: str = "input",
        tile_size: int = DEFAULT_TILE_SIZE,
        num_samples: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.input_name = input_name
        self.tile_size = tile_size
        self.idx = 0

        if image_paths and len(image_paths) > 0:
            self.image_paths = [str(p) for p in image_paths[:num_samples]]
            self.use_synthetic = False
            logger.info(
                "Initialized AerialCalibrationDataReader with %d real images (tile_size=%d)",
                len(self.image_paths),
                self.tile_size,
            )
        else:
            self.num_samples = num_samples
            self.use_synthetic = True
            self.rng = np.random.default_rng(seed)
            logger.warning(
                "No calibration image paths provided. Falling back to %d synthetic normalized tensors.",
                num_samples,
            )

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.use_synthetic:
            if self.idx >= len(self.image_paths):
                return None
            img_path = self.image_paths[self.idx]
            self.idx += 1
            try:
                tensor = preprocess_image(img_path, target_size=(self.tile_size, self.tile_size))
                return {self.input_name: tensor}
            except Exception as e:
                logger.warning("Failed to preprocess calibration image '%s': %s. Generating random tile.", img_path, e)
                fallback = np.random.randn(1, 3, self.tile_size, self.tile_size).astype(np.float32)
                return {self.input_name: fallback}
        else:
            if self.idx >= self.num_samples:
                return None
            self.idx += 1
            # Generate synthetic aerial-like normalized data with Gaussian noise
            synth_tensor = self.rng.standard_normal((1, 3, self.tile_size, self.tile_size), dtype=np.float32)
            return {self.input_name: synth_tensor}

    def rewind(self):
        self.idx = 0


# ==============================================================================
# Graph Preprocessing (Shape Inference & Optimization)
# ==============================================================================

def preprocess_onnx_graph(
    input_model_path: str,
    output_model_path: Optional[str] = None,
) -> str:
    """
    Applies ONNX Runtime's `quant_pre_process` to run shape inference, constant folding,
    and operator graph transformations before quantization.
    """
    if output_model_path is None:
        p = Path(input_model_path)
        output_model_path = str(p.parent / f"{p.stem}.preprocessed{p.suffix}")

    logger.info("Running ONNX graph pre-processing on '%s' -> '%s'...", input_model_path, output_model_path)
    try:
        quant_pre_process(
            input_model=input_model_path,
            output_model_path=output_model_path,
            skip_optimization=False,
            auto_merge=True,
        )
        logger.info("Graph pre-processing completed successfully.")
        return output_model_path
    except Exception as err:
        logger.warning(
            "quant_pre_process encountered an error (%s). Proceeding with original model file.",
            err,
        )
        return input_model_path


# ==============================================================================
# Dynamic Quantization
# ==============================================================================

def find_depthwise_conv_nodes(model_path: str) -> List[str]:
    """Finds all depthwise convolution node names in an ONNX model (group > 1)."""
    try:
        model = onnx.load(model_path)
        nodes_to_exclude = []
        for n in model.graph.node:
            if n.op_type == "Conv":
                attrs = {a.name: a.i for a in n.attribute if a.name == "group"}
                if attrs.get("group", 1) > 1:
                    nodes_to_exclude.append(n.name)
        return nodes_to_exclude
    except Exception as e:
        logger.warning("Could not scan for depthwise convolutions: %s", e)
        return []


def quantize_onnx_dynamic(
    model_input: str,
    model_output: str,
    weight_type: QuantType = QuantType.QUInt8,
    per_channel: bool = True,
    op_types_to_quantize: Optional[List[str]] = None,
    nodes_to_exclude: Optional[List[str]] = None,
    exclude_depthwise: bool = False,
    run_pre_process: bool = True,
) -> str:
    """
    Performs dynamic INT8 quantization on an ONNX model.
    Weights are quantized ahead of time, while activations are quantized dynamically during runtime.

    Args:
        model_input: Path to FP32 ONNX model.
        model_output: Path to save quantized ONNX model.
        weight_type: QuantType (QuantType.QUInt8 or QuantType.QInt8). Note that QUInt8
                     is the most compatible type for ConvInteger on CPUExecutionProvider.
        per_channel: Whether to quantize weights per-channel (recommended for segmentation accuracy).
        op_types_to_quantize: List of ONNX operator types to quantize (default: Conv, MatMul, Gemm).
        nodes_to_exclude: Specific node names to exclude from quantization.
        exclude_depthwise: Whether to automatically exclude depthwise convolutions (prevents latency regression).
        run_pre_process: Whether to run `quant_pre_process` first.

    Returns:
        Path to the quantized model output file.
    """
    if op_types_to_quantize is None:
        op_types_to_quantize = ["Conv", "MatMul", "Gemm"]

    os.makedirs(os.path.dirname(os.path.abspath(model_output)), exist_ok=True)

    temp_preprocessed: Optional[str] = None
    source_model = model_input

    if run_pre_process:
        temp_preprocessed = str(Path(model_output).parent / f"temp_{Path(model_input).stem}_preproc.onnx")
        source_model = preprocess_onnx_graph(model_input, temp_preprocessed)

    final_exclude = list(nodes_to_exclude or [])
    if exclude_depthwise:
        dw_nodes = find_depthwise_conv_nodes(source_model)
        if dw_nodes:
            logger.info("Auto-excluding %d depthwise convolution nodes from quantization to avoid memory-bound latency overhead.", len(dw_nodes))
            final_exclude = list(set(final_exclude + dw_nodes))

    logger.info(
        "Executing dynamic quantization:\n"
        "  - Input Model:          %s\n"
        "  - Output Model:         %s\n"
        "  - Weight Type:          %s\n"
        "  - Per-channel weights:  %s\n"
        "  - Quantized Op Types:   %s\n"
        "  - Excluded Nodes:       %d",
        source_model,
        model_output,
        weight_type.name,
        per_channel,
        op_types_to_quantize,
        len(final_exclude),
    )

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=source_model,
        model_output=model_output,
        per_channel=per_channel,
        weight_type=weight_type,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_exclude=final_exclude if final_exclude else None,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Dynamic quantization finished in %.2fs.", elapsed)

    # Clean up temporary preprocessed file if created
    if temp_preprocessed and os.path.exists(temp_preprocessed) and temp_preprocessed != model_input:
        try:
            os.remove(temp_preprocessed)
        except OSError:
            pass

    return model_output


# ==============================================================================
# Static Quantization
# ==============================================================================

def quantize_onnx_static(
    model_input: str,
    model_output: str,
    calibration_data_reader: CalibrationDataReader,
    quant_format: QuantFormat = QuantFormat.QDQ,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    per_channel: bool = False,
    op_types_to_quantize: Optional[List[str]] = None,
    run_pre_process: bool = True,
) -> str:
    """
    Performs static INT8 quantization using a calibration dataset.
    Both weights and activations are quantized to INT8.

    Args:
        model_input: Path to FP32 ONNX model.
        model_output: Path to save quantized ONNX model.
        calibration_data_reader: An instance of CalibrationDataReader providing calibration batches.
        quant_format: QuantFormat.QDQ (QuantizeLinear / DequantizeLinear) or QuantFormat.QOperator.
        weight_type: QuantType for weights (e.g. QInt8).
        activation_type: QuantType for activations (e.g. QInt8 or QUInt8).
        calibrate_method: CalibrationMethod (MinMax, Entropy, or Percentile).
        per_channel: Whether to quantize weights per-channel.
        op_types_to_quantize: Optional list of operator types to quantize.
        run_pre_process: Whether to run `quant_pre_process` first.

    Returns:
        Path to the quantized model output file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_output)), exist_ok=True)

    temp_preprocessed: Optional[str] = None
    source_model = model_input

    if run_pre_process:
        temp_preprocessed = str(Path(model_output).parent / f"temp_{Path(model_input).stem}_static_preproc.onnx")
        source_model = preprocess_onnx_graph(model_input, temp_preprocessed)

    logger.info(
        "Executing static quantization:\n"
        "  - Input Model:       %s\n"
        "  - Output Model:      %s\n"
        "  - Quant Format:      %s\n"
        "  - Weight Type:       %s\n"
        "  - Activation Type:   %s\n"
        "  - Calibrate Method:  %s\n"
        "  - Per-channel:       %s",
        source_model,
        model_output,
        quant_format.name,
        weight_type.name,
        activation_type.name,
        calibrate_method.name,
        per_channel,
    )

    t0 = time.perf_counter()
    quantize_static(
        model_input=source_model,
        model_output=model_output,
        calibration_data_reader=calibration_data_reader,
        quant_format=quant_format,
        weight_type=weight_type,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        per_channel=per_channel,
        op_types_to_quantize=op_types_to_quantize,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Static quantization finished in %.2fs.", elapsed)

    if temp_preprocessed and os.path.exists(temp_preprocessed) and temp_preprocessed != model_input:
        try:
            os.remove(temp_preprocessed)
        except OSError:
            pass

    return model_output


# ==============================================================================
# Model Verification & Benchmarking
# ==============================================================================

def verify_and_benchmark(
    fp32_model_path: str,
    quant_model_path: str,
    test_image_path: Optional[str] = None,
    num_warmup: int = 3,
    num_bench_runs: int = 10,
) -> Dict[str, Union[float, int, str]]:
    """
    Validates model integrity with `onnx.checker`, measures file compression,
    and runs comparative inference between FP32 and INT8 ONNX models using ONNX Runtime.

    Reports:
      - Model file size and compression factor
      - Max and Mean Absolute Error (MAE) on logits
      - Cosine similarity between output logits
      - Prediction label agreement rate (% identical argmax class predictions)
      - Latency benchmark comparison (FP32 ms vs INT8 ms and speedup factor)
    """
    logger.info("=" * 60)
    logger.info("Verifying Quantized Model: %s", quant_model_path)
    logger.info("=" * 60)

    # 1. Structural Check
    logger.info("1. Running ONNX model structure integrity check...")
    onnx_proto = onnx.load(quant_model_path)
    checker.check_model(onnx_proto)
    logger.info("   -> Model syntax and graph topology are valid.")

    # 2. File Size Comparison
    fp32_size_bytes = os.path.getsize(fp32_model_path)
    quant_size_bytes = os.path.getsize(quant_model_path)
    fp32_mb = fp32_size_bytes / (1024 * 1024)
    quant_mb = quant_size_bytes / (1024 * 1024)
    compression_ratio = fp32_size_bytes / max(quant_size_bytes, 1)

    logger.info("2. Disk Storage Comparison:")
    logger.info("   - FP32 Model Size:      %6.2f MB", fp32_mb)
    logger.info("   - INT8 Model Size:      %6.2f MB", quant_mb)
    logger.info("   - Compression Factor:   %6.2fx reduction (%.1f%% smaller)", compression_ratio, (1.0 - quant_size_bytes / fp32_size_bytes) * 100)

    # 3. Create Inference Sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session_fp32 = ort.InferenceSession(fp32_model_path, opts, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(quant_model_path, opts, providers=["CPUExecutionProvider"])

    input_name_fp32 = session_fp32.get_inputs()[0].name
    input_name_int8 = session_int8.get_inputs()[0].name

    # 4. Prepare Test Input Tensor
    if test_image_path and os.path.isfile(test_image_path):
        logger.info("3. Using real test image for validation: '%s'", test_image_path)
        input_tensor = preprocess_image(test_image_path, target_size=(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE))
    else:
        logger.info("3. Using synthetic normalized tile tensor (1, 3, 512, 512) for validation")
        rng = np.random.default_rng(42)
        input_tensor = rng.standard_normal((1, 3, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE), dtype=np.float32)

    # 5. Numerical Accuracy Evaluation
    out_fp32 = session_fp32.run(None, {input_name_fp32: input_tensor})[0]
    out_int8 = session_int8.run(None, {input_name_int8: input_tensor})[0]

    abs_diff = np.abs(out_fp32 - out_int8)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    # Cosine similarity across flattened vectors
    v_fp32 = out_fp32.reshape(-1).astype(np.float64)
    v_int8 = out_int8.reshape(-1).astype(np.float64)
    norm_fp32 = np.linalg.norm(v_fp32)
    norm_int8 = np.linalg.norm(v_int8)
    cos_sim = float(np.dot(v_fp32, v_int8) / (norm_fp32 * norm_int8)) if norm_fp32 > 0 and norm_int8 > 0 else 0.0

    # Class argmax agreement rate
    pred_fp32 = np.argmax(out_fp32, axis=1)
    pred_int8 = np.argmax(out_int8, axis=1)
    match_rate = float(np.mean(pred_fp32 == pred_int8) * 100.0)

    logger.info("4. Output Precision & Prediction Agreement:")
    logger.info("   - Output Shape:         %s", out_int8.shape)
    logger.info("   - Max Absolute Error:   %8.4f", max_diff)
    logger.info("   - Mean Absolute Error:  %8.4f", mean_diff)
    logger.info("   - Cosine Similarity:    %8.6f", cos_sim)
    logger.info("   - Class Match Rate:     %8.2f%% of pixels identical to FP32", match_rate)

    # 6. Latency Benchmark
    logger.info("5. Latency Benchmarking (intra_op_threads=4, CPUExecutionProvider):")
    # Warmup
    for _ in range(num_warmup):
        session_fp32.run(None, {input_name_fp32: input_tensor})
        session_int8.run(None, {input_name_int8: input_tensor})

    # Benchmark FP32
    fp32_times = []
    for _ in range(num_bench_runs):
        t_start = time.perf_counter()
        session_fp32.run(None, {input_name_fp32: input_tensor})
        fp32_times.append(time.perf_counter() - t_start)
    avg_fp32_ms = float(np.median(fp32_times) * 1000.0)

    # Benchmark INT8
    int8_times = []
    for _ in range(num_bench_runs):
        t_start = time.perf_counter()
        session_int8.run(None, {input_name_int8: input_tensor})
        int8_times.append(time.perf_counter() - t_start)
    avg_int8_ms = float(np.median(int8_times) * 1000.0)

    speedup = avg_fp32_ms / max(avg_int8_ms, 1e-4)

    logger.info("   - FP32 Median Latency:  %6.2f ms / tile", avg_fp32_ms)
    logger.info("   - INT8 Median Latency:  %6.2f ms / tile", avg_int8_ms)
    logger.info("   - Effective Speedup:    %6.2fx", speedup)
    logger.info("=" * 60)

    return {
        "fp32_model_path": fp32_model_path,
        "quant_model_path": quant_model_path,
        "fp32_size_mb": fp32_mb,
        "quant_size_mb": quant_mb,
        "compression_ratio": compression_ratio,
        "max_abs_error": max_diff,
        "mean_abs_error": mean_diff,
        "cosine_similarity": cos_sim,
        "class_match_rate_pct": match_rate,
        "fp32_latency_ms": avg_fp32_ms,
        "int8_latency_ms": avg_int8_ms,
        "speedup": speedup,
    }


# ==============================================================================
# High-Level Orchestrator
# ==============================================================================

def quantize_onnx(
    input_model: str = DEFAULT_INPUT_MODEL,
    output_model: Optional[str] = None,
    mode: str = "dynamic",
    weight_type_str: str = "auto",
    activation_type_str: str = "qint8",
    calibration_data_dir: Optional[str] = None,
    num_calibration_samples: int = 32,
    per_channel: bool = True,
    run_pre_process: bool = True,
    verify: bool = True,
    test_image: Optional[str] = None,
) -> Dict[str, str]:
    """
    Main entrypoint to perform ONNX model quantization with configurable mode:
      - 'dynamic': Quantizes weights ahead of time; activations are quantized at runtime.
      - 'static': Quantizes weights and activations using a calibration dataset.
      - 'both' / 'all': Generates both dynamic and static quantized models for direct comparison.

    Returns:
        Dictionary mapping mode name to output model file path.
    """
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input ONNX model not found at '{input_model}'.")

    results: Dict[str, str] = {}
    mode = mode.lower().strip()

    if test_image is None:
        sample_images = find_default_calibration_images(limit=1)
        if sample_images:
            test_image = sample_images[0]
            logger.info("Auto-selected test image for verification: '%s'", test_image)

    # Determine Weight Type
    def resolve_quant_type(val: str, default: QuantType) -> QuantType:
        val_l = val.lower().strip()
        if val_l in ("quint8", "uint8", "u8"):
            return QuantType.QUInt8
        elif val_l in ("qint8", "int8", "i8"):
            return QuantType.QInt8
        return default

    # 0. Partial / Selective Mode (Optimized for real latency speedups)
    if mode in ("partial", "selective"):
        part_out = output_model
        if part_out is None:
            inp_p = Path(input_model)
            part_out = str(inp_p.parent / f"{inp_p.stem}_partial_int8{inp_p.suffix}")

        proto = onnx.load(input_model)
        op_types = set(n.op_type for n in proto.graph.node)
        has_matmul = "MatMul" in op_types or "Gemm" in op_types
        has_layernorm = "LayerNormalization" in op_types

        if has_matmul and has_layernorm:
            # Transformer model (SegFormer / ViT):
            # Quantize ONLY MatMul & Gemm, leaving Convs, LayerNorms, Reshapes untouched
            logger.info("Detected Vision Transformer (SegFormer). Quantizing MatMul/Gemm layers only.")
            w_type = resolve_quant_type(weight_type_str, default=QuantType.QInt8)
            quantize_onnx_dynamic(
                model_input=input_model,
                model_output=part_out,
                weight_type=w_type,
                per_channel=per_channel,
                op_types_to_quantize=["MatMul", "Gemm"],
                run_pre_process=False,
            )
        else:
            # Hybrid / Conv model (FastViT / EfficientNet):
            # Quantize standard dense Convs, automatically excluding all depthwise convs
            logger.info("Detected CNN / Hybrid model. Quantizing dense convolutions while excluding depthwise layers.")
            w_type = resolve_quant_type(weight_type_str, default=QuantType.QUInt8)
            quantize_onnx_dynamic(
                model_input=input_model,
                model_output=part_out,
                weight_type=w_type,
                per_channel=per_channel,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                exclude_depthwise=True,
                run_pre_process=run_pre_process,
            )
        results["partial"] = part_out

        if verify:
            verify_and_benchmark(input_model, part_out, test_image_path=test_image)

    # 1. Dynamic Mode
    if mode in ("dynamic", "both", "all"):
        dyn_out = output_model
        if dyn_out is None or mode in ("both", "all"):
            dyn_out = DEFAULT_DYNAMIC_OUTPUT_MODEL if mode in ("both", "all") else (output_model or DEFAULT_DYNAMIC_OUTPUT_MODEL)

        w_type = resolve_quant_type(weight_type_str, default=QuantType.QUInt8)
        quantize_onnx_dynamic(
            model_input=input_model,
            model_output=dyn_out,
            weight_type=w_type,
            per_channel=per_channel,
            run_pre_process=run_pre_process,
        )
        results["dynamic"] = dyn_out

        if verify:
            verify_and_benchmark(input_model, dyn_out, test_image_path=test_image)

    # 2. Static Mode
    if mode in ("static", "both", "all"):
        stat_out = output_model
        if stat_out is None or mode in ("both", "all"):
            stat_out = DEFAULT_STATIC_OUTPUT_MODEL if mode in ("both", "all") else (output_model or DEFAULT_STATIC_OUTPUT_MODEL)

        # Collect calibration images
        calib_images: List[str] = []
        if calibration_data_dir and os.path.isdir(calibration_data_dir):
            patterns = [
                os.path.join(calibration_data_dir, "*.png"),
                os.path.join(calibration_data_dir, "*.jpg"),
                os.path.join(calibration_data_dir, "*.jpeg"),
            ]
            for p in patterns:
                calib_images.extend(glob.glob(p))
            calib_images.sort()
            calib_images = calib_images[:num_calibration_samples]
            logger.info("Found %d calibration images in '%s'", len(calib_images), calibration_data_dir)
        elif calibration_data_dir and os.path.isfile(calibration_data_dir):
            calib_images = [calibration_data_dir]
        else:
            calib_images = find_default_calibration_images(limit=num_calibration_samples)

        data_reader = AerialCalibrationDataReader(
            image_paths=calib_images if calib_images else None,
            tile_size=DEFAULT_TILE_SIZE,
            num_samples=num_calibration_samples,
        )

        w_type = resolve_quant_type(weight_type_str, default=QuantType.QInt8)
        a_type = resolve_quant_type(activation_type_str, default=QuantType.QInt8)

        quantize_onnx_static(
            model_input=input_model,
            model_output=stat_out,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=w_type,
            activation_type=a_type,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=False,  # QDQ standard
            run_pre_process=run_pre_process,
        )
        results["static"] = stat_out

        if verify:
            verify_and_benchmark(input_model, stat_out, test_image_path=test_image)

    return results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize ONNX aerial semantic segmentation model to INT8/UINT8 precision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-model",
        type=str,
        default=DEFAULT_INPUT_MODEL,
        help="Path to the input FP32 ONNX model file.",
    )
    parser.add_argument(
        "-o",
        "--output-model",
        type=str,
        default=None,
        help="Path for saving quantized ONNX model. If omitted, default path based on mode is used.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="partial",
        choices=["partial", "dynamic", "static", "both", "all"],
        help="Quantization mode: 'partial' (smart selective quantization for real latency speedup), 'dynamic', 'static', or 'both'.",
    )
    parser.add_argument(
        "--exclude-depthwise",
        action="store_true",
        help="Exclude depthwise convolutions from quantization to prevent memory-bound latency penalty.",
    )
    parser.add_argument(
        "-w",
        "--weight-type",
        type=str,
        default="auto",
        choices=["auto", "quint8", "qint8"],
        help="Weight precision: 'quint8' (unsigned 8-bit, best for CPU ConvInteger) or 'qint8' (signed 8-bit).",
    )
    parser.add_argument(
        "-a",
        "--activation-type",
        type=str,
        default="qint8",
        choices=["quint8", "qint8"],
        help="Activation precision for static quantization.",
    )
    parser.add_argument(
        "-c",
        "--calibrate-data",
        type=str,
        default=None,
        help="Directory or file path of calibration images for static quantization. Auto-detected if omitted.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=32,
        help="Number of calibration image samples for static quantization.",
    )
    parser.add_argument(
        "--per-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-channel (channel-wise) weight quantization for higher accuracy.",
    )
    parser.add_argument(
        "--pre-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX shape inference and graph optimization prior to quantization.",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate model integrity and print latency/accuracy comparison against FP32.",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="Optional test image path used for verification benchmarks.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting ONNX Model Quantization Pipeline...")

    try:
        quantize_onnx(
            input_model=args.input_model,
            output_model=args.output_model,
            mode=args.mode,
            weight_type_str=args.weight_type,
            activation_type_str=args.activation_type,
            calibration_data_dir=args.calibrate_data,
            num_calibration_samples=args.num_samples,
            per_channel=args.per_channel,
            run_pre_process=args.pre_process,
            verify=args.verify,
            test_image=args.test_image,
        )
        logger.info("Quantization completed successfully!")
    except Exception as exc:
        logger.exception("Quantization failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
