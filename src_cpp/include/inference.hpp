#pragma once

#include <string>
#include <vector>
#include <utility>
#include "types.hpp"
#include "onnx_model.hpp"

namespace segmentation {

std::vector<size_t> tile_starts(size_t length, size_t tile_size, size_t stride);

// Loads an image file using stb_image, converts to RGB bytes.
std::vector<uint8_t> load_image_rgb(const std::string& filepath, int& width, int& height);

// Loads an image file using stb_image preserving native channels.
std::vector<uint8_t> load_image_raw(const std::string& filepath, int& width, int& height, int& channels);

// Saves RGB buffer as PNG using stb_image_write.
void save_png(const std::string& filepath, int width, int height, const uint8_t* rgb_data);

// Full image segmentation with sliding-window tiling and timing.
std::pair<std::vector<uint8_t>, InferenceTiming> predict_full_resolution_with_timing(
    OnnxModel& model,
    const uint8_t* rgb_bytes,
    size_t width,
    size_t height,
    const TilingConfig& config
);

// Map class ID (0..5) to RGB color mask.
std::vector<uint8_t> render_class_mask_to_rgb(const std::vector<uint8_t>& class_mask);

// Decode ground-truth mask buffer into class IDs (0..5).
std::vector<uint8_t> decode_ground_truth_mask(
    const uint8_t* raw_bytes,
    size_t width,
    size_t height,
    int channels
);

// Calculate IoU per class and mean IoU.
ClassIouResult calculate_mask_iou(
    const uint8_t* prediction,
    const uint8_t* target,
    size_t num_pixels
);

} // namespace segmentation

