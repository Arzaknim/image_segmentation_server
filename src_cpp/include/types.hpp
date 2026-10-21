#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <array>
#include <cmath>
#include <optional>
#include <limits>

namespace segmentation {

constexpr size_t NUM_CLASSES = 6;
constexpr uint8_t IGNORE_INDEX = 5;
constexpr size_t DEFAULT_TILE_SIZE = 512;
constexpr size_t DEFAULT_TILE_STRIDE = 256;

constexpr std::array<float, 3> IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> IMAGENET_STD = {0.229f, 0.224f, 0.225f};

const std::array<std::string, NUM_CLASSES> CLASS_NAMES = {
    "Water",
    "Land (unpaved area)",
    "Road",
    "Building",
    "Vegetation",
    "Unlabeled"
};

const std::array<std::array<uint8_t, 3>, NUM_CLASSES> CLASS_COLORS_RGB = {{
    {0xE2, 0xA9, 0x29}, // 0: Water (#E2A929)
    {0x84, 0x29, 0xF6}, // 1: Land (#8429F6)
    {0x6E, 0xC1, 0xE4}, // 2: Road (#6EC1E4)
    {0x3C, 0x10, 0x98}, // 3: Building (#3C1098)
    {0xFE, 0xDD, 0x3A}, // 4: Vegetation (#FEDD3A)
    {0x9B, 0x9B, 0x9B}  // 5: Unlabeled (#9B9B9B)
}};

struct TilingConfig {
    size_t tile_size = DEFAULT_TILE_SIZE;
    size_t stride = DEFAULT_TILE_STRIDE;
};

struct ClassIouResult {
    std::array<double, NUM_CLASSES - 1> per_class_iou{}; // std::nan if union == 0
    double mean_iou = 0.0;
    std::array<uint64_t, NUM_CLASSES - 1> intersections{};
    std::array<uint64_t, NUM_CLASSES - 1> unions{};
};

struct InferenceTiming {
    std::chrono::duration<double> total_inference_time{0.0};
    std::chrono::duration<double> model_forward_time{0.0};
    size_t tile_count = 0;

    double avg_tile_forward_ms() const {
        if (tile_count > 0) {
            return (model_forward_time.count() * 1000.0) / static_cast<double>(tile_count);
        }
        return 0.0;
    }
};

} // namespace segmentation

