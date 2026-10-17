#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include "onnxruntime_cxx_api.h"
#include "types.hpp"

namespace segmentation {

class OnnxModel {
public:
    explicit OnnxModel(const std::string& model_path);
    ~OnnxModel() = default;

    OnnxModel(const OnnxModel&) = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;
    OnnxModel(OnnxModel&&) = default;
    OnnxModel& operator=(OnnxModel&&) = default;

    // Run inference on a planar [channels, height, width] float buffer.
    // Returns [num_classes, height, width] planar logits.
    std::vector<float> predict_tile(const float* tile_data, size_t channels, size_t height, size_t width);

    size_t get_num_classes() const { return num_classes_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<const char*> input_names_ptrs_;
    std::vector<const char*> output_names_ptrs_;

    size_t num_classes_ = NUM_CLASSES;
    mutable std::mutex session_mutex_;
};

} // namespace segmentation

