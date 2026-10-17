#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "inference.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>

namespace segmentation {

std::vector<size_t> tile_starts(size_t length, size_t tile_size, size_t stride) {
    if (tile_size == 0 || stride == 0) {
        throw std::invalid_argument("tile_size and stride must be positive");
    }
    if (length <= tile_size) {
        return {0};
    }

    std::vector<size_t> starts;
    for (size_t s = 0; s <= length - tile_size; s += stride) {
        starts.push_back(s);
    }
    size_t final_start = length - tile_size;
    if (starts.empty() || starts.back() != final_start) {
        starts.push_back(final_start);
    }
    return starts;
}

std::vector<uint8_t> load_image_rgb(const std::string& filepath, int& width, int& height) {
    int channels_in_file = 0;
    // Force 3 channels (RGB)
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels_in_file, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image at '" + filepath + "': " + stbi_failure_reason());
    }

    size_t total_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    std::vector<uint8_t> rgb_bytes(data, data + total_bytes);
    stbi_image_free(data);
    return rgb_bytes;
}

void save_png(const std::string& filepath, int width, int height, const uint8_t* rgb_data) {
    int stride_in_bytes = width * 3;
    int success = stbi_write_png(filepath.c_str(), width, height, 3, rgb_data, stride_in_bytes);
    if (!success) {
        throw std::runtime_error("Failed to write PNG image to '" + filepath + "'");
    }
}

std::pair<std::vector<uint8_t>, InferenceTiming> predict_full_resolution_with_timing(
    OnnxModel& model,
    const uint8_t* rgb_bytes,
    size_t width,
    size_t height,
    const TilingConfig& config
) {
    auto total_start = std::chrono::high_resolution_clock::now();

    InferenceTiming timing;
    timing.tile_count = 0;

    std::vector<float> probability_canvas(NUM_CLASSES * height * width, 0.0f);
    std::vector<float> count_canvas(height * width, 0.0f);

    auto y_starts = tile_starts(height, config.tile_size, config.stride);
    auto x_starts = tile_starts(width, config.tile_size, config.stride);

    size_t tile_plane_size = config.tile_size * config.tile_size;
    std::vector<float> tile_input(3 * tile_plane_size, 0.0f);

    for (size_t y_start : y_starts) {
        size_t tile_h = std::min(config.tile_size, height - y_start);
        for (size_t x_start : x_starts) {
            size_t tile_w = std::min(config.tile_size, width - x_start);

            // Zero-fill input buffer for boundary padding
            std::fill(tile_input.begin(), tile_input.end(), 0.0f);

            // Normalize RGB into [3, tile_size, tile_size] CHW layout
            for (size_t ty = 0; ty < tile_h; ++ty) {
                size_t iy = y_start + ty;
                for (size_t tx = 0; tx < tile_w; ++tx) {
                    size_t ix = x_start + tx;
                    size_t pixel_offset = (iy * width + ix) * 3;
                    for (size_t c = 0; c < 3; ++c) {
                        float raw = static_cast<float>(rgb_bytes[pixel_offset + c]) / 255.0f;
                        float norm = (raw - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                        tile_input[c * tile_plane_size + ty * config.tile_size + tx] = norm;
                    }
                }
            }

            // Model forward pass timing
            auto fwd_start = std::chrono::high_resolution_clock::now();
            std::vector<float> logits = model.predict_tile(
                tile_input.data(), 3, config.tile_size, config.tile_size
            );
            auto fwd_end = std::chrono::high_resolution_clock::now();
            timing.model_forward_time += (fwd_end - fwd_start);
            timing.tile_count++;

            // Softmax along channels for the valid region [0, tile_h) x [0, tile_w)
            for (size_t ty = 0; ty < tile_h; ++ty) {
                for (size_t tx = 0; tx < tile_w; ++tx) {
                    float max_val = -1e30f;
                    for (size_t c = 0; c < NUM_CLASSES; ++c) {
                        float val = logits[c * tile_plane_size + ty * config.tile_size + tx];
                        if (val > max_val) max_val = val;
                    }

                    float sum_exp = 0.0f;
                    float exps[NUM_CLASSES];
                    for (size_t c = 0; c < NUM_CLASSES; ++c) {
                        float val = logits[c * tile_plane_size + ty * config.tile_size + tx];
                        exps[c] = std::exp(val - max_val);
                        sum_exp += exps[c];
                    }

                    size_t canvas_idx = (y_start + ty) * width + (x_start + tx);
                    count_canvas[canvas_idx] += 1.0f;

                    for (size_t c = 0; c < NUM_CLASSES; ++c) {
                        probability_canvas[c * (height * width) + canvas_idx] += (exps[c] / sum_exp);
                    }
                }
            }
        }
    }

    // Normalize probabilities and compute argmax
    std::vector<uint8_t> mask(height * width, 0);
    for (size_t i = 0; i < height * width; ++i) {
        float count = std::max(1.0f, count_canvas[i]);
        float max_p = -1.0f;
        uint8_t best_c = 0;
        for (size_t c = 0; c < NUM_CLASSES; ++c) {
            float p = (probability_canvas[c * (height * width) + i] /= count);
            if (p > max_p) {
                max_p = p;
                best_c = static_cast<uint8_t>(c);
            }
        }
        mask[i] = best_c;
    }

    timing.total_inference_time = std::chrono::high_resolution_clock::now() - total_start;
    return {mask, timing};
}

std::vector<uint8_t> render_class_mask_to_rgb(const std::vector<uint8_t>& class_mask) {
    std::vector<uint8_t> rgb;
    rgb.reserve(class_mask.size() * 3);
    for (uint8_t cls : class_mask) {
        size_t c_idx = (cls < NUM_CLASSES) ? cls : (NUM_CLASSES - 1);
        const auto& color = CLASS_COLORS_RGB[c_idx];
        rgb.push_back(color[0]);
        rgb.push_back(color[1]);
        rgb.push_back(color[2]);
    }
    return rgb;
}

} // namespace segmentation

